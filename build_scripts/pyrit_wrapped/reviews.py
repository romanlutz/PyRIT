# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json

from pydantic import AwareDatetime, BaseModel, Field

from build_scripts.pyrit_wrapped.github_client import GitHubClient, object_data
from build_scripts.pyrit_wrapped.models import Actor, Review, WrappedError


class GraphActor(BaseModel):
    id: str
    login: str
    type: str = Field(alias="__typename")
    database_id: int | None = Field(alias="databaseId", default=None)

    def to_actor(self) -> Actor:
        return Actor(id=self.id, login=self.login, type=self.type, database_id=self.database_id)


class GraphReview(BaseModel):
    id: int = Field(alias="fullDatabaseId")
    author: GraphActor | None
    state: str
    submitted_at: AwareDatetime | None = Field(alias="submittedAt")
    url: str
    has_body: bool = Field(alias="_wrapped_has_body")

    def to_review(self, number: int) -> Review:
        return Review(
            id=self.id,
            item_number=number,
            author=self.author.to_actor() if self.author else None,
            state=self.state,
            submitted_at=self.submitted_at,
            url=self.url,
            has_body=self.has_body,
        )


class PageInfo(BaseModel):
    has_next: bool = Field(alias="hasNextPage")
    cursor: str | None = Field(alias="endCursor")


class ReviewConnection(BaseModel):
    nodes: list[GraphReview]
    page_info: PageInfo = Field(alias="pageInfo")


class GraphPull(BaseModel):
    reviews: ReviewConnection


class ReviewReader:
    def __init__(self, client: GitHubClient) -> None:
        self.client = client

    async def read_async(self, numbers: list[int]) -> list[Review]:
        result: dict[int, Review] = {}
        for offset in range(0, len(numbers), 20):
            pending: dict[int, str | None] = dict.fromkeys(numbers[offset : offset + 20])
            seen: dict[int, set[str]] = {number: set() for number in pending}
            while pending:
                response = await self.client.get_async(path="graphql", params={"query": self._query(pending)})
                data = object_data(response.data)
                repository = object_data(object_data(data["data"])["repository"])
                following: dict[int, str | None] = {}
                for number in pending:
                    value = repository.get(f"p{number}")
                    if value is None:
                        raise WrappedError(f"Review query omitted PR {number}; cannot claim complete activity.")
                    connection = GraphPull.model_validate(value).reviews
                    for item in connection.nodes:
                        result[item.id] = item.to_review(number)
                    if connection.page_info.has_next:
                        cursor = connection.page_info.cursor
                        if cursor is None or cursor in seen[number]:
                            raise WrappedError(f"Nonadvancing GraphQL reviews for PR {number}.")
                        seen[number].add(cursor)
                        following[number] = cursor
                pending = following
            self.client.progress(f"Read reviews for {min(offset + 20, len(numbers))}/{len(numbers)} PRs.")
        return sorted(result.values(), key=lambda review: review.id)

    @staticmethod
    def _query(pending: dict[int, str | None]) -> str:
        fields = []
        for number, cursor in sorted(pending.items()):
            fields.append(
                f"p{number}: pullRequest(number:{number}) "
                "{ reviews(first:100, after:" + json.dumps(cursor) + ") "
                "{ nodes { fullDatabaseId state submittedAt url body author "
                "{ login __typename ... on Node { id } ... on User { databaseId } ... on Bot { databaseId } } } "
                "pageInfo { hasNextPage endCursor } } }"
            )
        return 'query { repository(owner:"microsoft", name:"PyRIT") { ' + " ".join(fields) + " } }"
