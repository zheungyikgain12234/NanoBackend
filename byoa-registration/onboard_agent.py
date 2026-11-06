"""
onboard_agent.py
~~~~~~~~~~~~~~~~~
Run this LOCALLY, on the platform owner's own machine/access, to onboard
a new agent -- yours or a third party's.

Deliberately a local script, not an HTTP endpoint: onboarding decides
WHO is allowed to claim an identity on the platform -- that's a judgment
call a human makes, not something that should be exposed over the
network for anyone to hit (that would just move the same trust problem
one level up, requiring yet another secret to protect an /onboard route).

Usage:
    python onboard_agent.py <agent_id> [--owner "Acme Corp"]

The secret is printed ONCE. It is not stored anywhere else, and there
is no way to retrieve it again -- if it's lost, revoke and re-onboard.
"""
import sys
import argparse

from agent_credentials import init_credentials_table, onboard_agent


def main():
    parser = argparse.ArgumentParser(description="Onboard a new agent onto the platform.")
    parser.add_argument("agent_id", help="The agent_id this party will register under, e.g. 'acme-invoice-agent'")
    parser.add_argument("--owner", default="", help="A human-readable label, e.g. 'Acme Corp' or 'internal'")
    args = parser.parse_args()

    init_credentials_table()

    try:
        secret = onboard_agent(args.agent_id, owner_label=args.owner)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(f"Onboarded: {args.agent_id}" + (f"  (owner: {args.owner})" if args.owner else ""))
    print("=" * 70)
    print()
    print("Secret (copy this now -- it will not be shown again):")
    print()
    print(f"    {secret}")
    print()
    print("Give this to the agent's owner. It must be set as BYOA_SECRET")
    print("in that agent's own .env file, and used nowhere else.")
    print("=" * 70)


if __name__ == "__main__":
    main()
