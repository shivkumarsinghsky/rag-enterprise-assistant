# ADR-002: Authorization Enforced in Retrieval, Scope From the Credential

- **Status:** Accepted
- **Date:** 2026-10-01

## Context

An assistant over enterprise documents must never reveal content a user cannot open in the source system. Asking
the model to "only use documents the user is allowed to see" is not a control — the model cannot enforce it, and
prompt injection can subvert it.

## Decision

- Every chunk stores `tenant` and `acl` (allowed groups) copied from the source at ingestion.
- The principal (tenant, user, groups) comes from the authenticated credential.
- Tenant and ACL predicates are part of every vector and keyword search; the request can only add narrowing
  filters (metadata, document ids).
- Conversation history is keyed by (tenant, user, conversation id).

## Alternatives Considered

- **Post-filtering retrieved results** — can return empty results when the top-k are forbidden, and leaks timing and
  existence information.
- **Index per user** — perfect isolation, unmanageable duplication.
- **Index per tenant** — stronger tenant isolation; still needs ACL filtering inside the tenant. A valid option for
  regulated tenants.

## Trade-offs

ACLs must be kept in sync with source systems; a permission change requires re-ingestion or an ACL update.

## Consequences

Tests verify that forbidden documents are never retrieved and that cross-tenant questions are refused.
