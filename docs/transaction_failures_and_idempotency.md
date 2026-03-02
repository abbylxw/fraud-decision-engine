# Transaction Failures & Idempotency

## Failure Categories

### 1. Client cancellation
The client drops the connection mid-request (timeout, network drop). The server may
have already processed and written the transaction. The client never receives the
response and retries with the same idempotency key — correct behaviour relies on
the idempotency layer returning the cached result.

### 2. Network failures
Response is computed and written to DB, but the network fails before the response
reaches the client. Identical to client cancellation from a retry perspective.

### 3. Downstream timeouts
Redis or PostgreSQL is slow or unreachable during processing. The request fails
mid-way — partially processed state is possible (e.g., DB written, velocity counters
not updated). Idempotency key may or may not have been set.

### 4. Partial failures
Velocity counter update (`update_velocity_counters`) runs fire-and-forget via
`asyncio.create_task`. If the server crashes between returning the response and the
task completing, counters are left stale — the next request for the same user sees
lower-than-actual velocity counts.

---

## Idempotency Handling Options

### Option 1: Idempotency key (current approach, partially)
Client sends a unique `Idempotency-Key` header. Server caches the response in Redis
for 24h keyed on it. Repeat requests with the same key return the cached response
without reprocessing.

**Gap:** Current GET→check→SET has a TOCTOU race (see Trade-off #5).

### Option 2: Atomic `SET NX EX` claim
Replace GET→check→SET with a single atomic `SET NX EX` that writes a sentinel value.
Only the first request to claim the key proceeds; subsequent requests poll until the
real response replaces the sentinel, then return it.

**Status:** Recommended fix — see `main.py` TODO.

### Option 3: Outbox pattern
Write the transaction result and the idempotency key atomically to PostgreSQL in a
single transaction. A background worker publishes the result to Redis. Guarantees
exactly-once DB write even if the server crashes mid-response.

**Trade-off:** Adds infra complexity; overkill for current scale.

### Option 4: At-least-once + idempotent inserts
Use `INSERT ... ON CONFLICT DO NOTHING` for DB writes so duplicate processing is
harmless. Combine with idempotency key for the response layer.

**Status:** Useful companion to Option 2.

### Option 5: Saga / compensating transactions
For multi-step workflows where partial failure needs rollback. Not applicable at
current scope (single DB write + Redis updates).

---

## Practical Priority

1. Fix TOCTOU with `SET NX` (Option 2) — highest impact, low effort
2. Add `ON CONFLICT DO NOTHING` to DB insert (Option 4) — safety net
3. Log and alert on stale velocity counter scenarios (partial failure monitoring)
