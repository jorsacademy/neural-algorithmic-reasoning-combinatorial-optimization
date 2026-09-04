from narcopt.cli import main

raise SystemExit(main(["train", *(__import__("sys").argv[1:])]))
