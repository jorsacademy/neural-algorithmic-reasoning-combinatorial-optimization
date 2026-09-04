from narcopt.cli import main

raise SystemExit(main(["collect", *(__import__("sys").argv[1:])]))
