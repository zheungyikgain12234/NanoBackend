"""
Create an admin login for the BYOA admin app. Run LOCALLY on the host:

    python create_admin.py <username>

The password is typed at a hidden prompt, never passed on the command line.
"""
import argparse
import getpass
import re
import sys

from dotenv import load_dotenv

load_dotenv()

from admin_users import create_admin, init_admin_table  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Create a BYOA admin user.")
    p.add_argument("username")
    args = p.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]{3,64}", args.username):
        sys.exit("ERROR: username must be 3-64 chars: letters, digits, . _ -")
    pw = getpass.getpass("Password (min 12 chars): ")
    if len(pw) < 12:
        sys.exit("ERROR: password must be at least 12 characters.")
    if pw != getpass.getpass("Repeat password: "):
        sys.exit("ERROR: passwords do not match.")
    init_admin_table()
    try:
        create_admin(args.username, pw)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Admin {args.username!r} created.")


if __name__ == "__main__":
    main()
