#!/usr/bin/env python
"""Live V3 validator entrypoint.

Obtain a public problem from the private source, query miners, commit their
    signed responses, retrieve the verifier, grade isolated miner artifacts,
    compute local scores, and set the validator's own weights.

Prerequisites (see scripts/register_testnet.sh):
  - a funded coldkey+hotkey, registered on your subnet's NETUID
  - .env set: NETUID, SUBTENSOR_NETWORK, WALLET_NAME, WALLET_HOTKEY
    and PROBLEM_SERVER_URL

    . .venv/bin/activate && PYTHONPATH=. python scripts/run_validator.py
"""

from rlvr.config import (
    get_settings,
    ignored_release_policy_keys,
    nondefault_settings_summary,
    release_policy_summary,
)


def main() -> None:
    settings = get_settings()
    if not settings.problem_server_url:
        raise SystemExit("set PROBLEM_SERVER_URL for the V3 problem source")

    ignored = ignored_release_policy_keys()
    if ignored:
        print(
            "[validator] WARN: ignored release-policy overrides: "
            + ", ".join(ignored)
        )
    print(f"[validator] release policy {release_policy_summary()}")
    nondefault = nondefault_settings_summary(settings)
    if nondefault:
        print(f"[validator] non-default machine settings {nondefault}")

    from rlvr.neurons.decentralized import run_decentralized_validator

    run_decentralized_validator(settings)


if __name__ == "__main__":
    main()
