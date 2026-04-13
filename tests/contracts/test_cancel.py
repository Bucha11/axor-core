"""Tests for CancelToken."""
from __future__ import annotations

import asyncio
import pytest
from axor_core.contracts.cancel import CancelToken, CancelReason, make_token


class TestCancelToken:

    def test_not_cancelled_by_default(self):
        token = make_token()
        assert not token.is_cancelled()
        assert token.reason is None

    def test_cancel_sets_flag(self):
        token = make_token()
        token.cancel(CancelReason.USER_ABORT)
        assert token.is_cancelled()
        assert token.reason == CancelReason.USER_ABORT

    def test_cancel_with_detail(self):
        token = make_token()
        token.cancel(CancelReason.BUDGET_EXHAUSTED, detail="spent 96%")
        assert token.detail == "spent 96%"

    def test_idempotent_first_reason_wins(self):
        token = make_token()
        token.cancel(CancelReason.USER_ABORT, detail="first")
        token.cancel(CancelReason.BUDGET_EXHAUSTED, detail="second")
        assert token.reason == CancelReason.USER_ABORT
        assert token.detail == "first"

    def test_child_token_independent(self):
        parent = make_token()
        child = parent.child_token()
        parent.cancel(CancelReason.USER_ABORT)
        assert parent.is_cancelled()
        assert not child.is_cancelled()

    def test_child_can_be_cancelled_independently(self):
        parent = make_token()
        child = parent.child_token()
        child.cancel(CancelReason.PARENT_CANCELLED)
        assert child.is_cancelled()
        assert not parent.is_cancelled()

    @pytest.mark.asyncio
    async def test_wait_resolves_on_cancel(self):
        token = make_token()

        async def cancel_soon():
            await asyncio.sleep(0.05)
            token.cancel(CancelReason.TIMEOUT, detail="timed out")

        asyncio.create_task(cancel_soon())
        reason = await token.wait()
        assert reason == CancelReason.TIMEOUT

    def test_all_cancel_reasons_defined(self):
        reasons = {r.value for r in CancelReason}
        assert "user_abort" in reasons
        assert "budget_exhausted" in reasons
        assert "parent_cancelled" in reasons
        assert "policy_denied" in reasons
        assert "timeout" in reasons
