from narcopt.cli import main

raise SystemExit(main(["research", *(__import__("sys").argv[1:])]))
