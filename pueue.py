from typing import Union, Optional  # , Tuple
import subprocess
import os
import sys
from cli import ResumeArgs
from train import TrainArgs
from loguru import logger

curr_dir = os.path.dirname(os.path.abspath(__file__))


def pueue_set_parallel(
    pueue_group: Optional[str] = None, pueue_parallel: Optional[int] = 1
) -> None:
    if pueue_group:
        temp_return = subprocess.run(
            f'pueue group add "{pueue_group}"', capture_output=True
        )
        logger.info(
            f"Create pueue group {pueue_group}: {temp_return.stdout.decode().strip()}"
        )

    if pueue_parallel > 1:
        if pueue_group:
            temp_return = subprocess.run(
                f'pueue parallel -g "{pueue_group}" {pueue_parallel}',
                capture_output=True,
            )
            logger.info(
                f"Set parallel for {pueue_group}: {temp_return.stdout.decode().strip()}"
            )
        else:
            temp_return = subprocess.run(
                f'pueue parallel {pueue_parallel}"', capture_output=True
            )
            logger.info(f"Set parallel: {temp_return.stdout.decode().strip()}")


def pueue_submit(
    parsed_args: Union[TrainArgs, ResumeArgs],
    pueue_group: Optional[str] = None,
    pueue_parallel: Optional[int] = 1,
    dir_path: str = curr_dir,
    dry_run: bool = False,
) -> str:  # Tuple[str, str]:
    args = ["pueue", "add"]

    if pueue_group:
        args.extend(["-g", f'"{pueue_group}"'])

    pueue_set_parallel(pueue_group, pueue_parallel)

    args.append("--")

    if isinstance(parsed_args, TrainArgs) or isinstance(parsed_args, ResumeArgs):
        # args.append("python")
        args.append(sys.executable)
        args.append("cli.py")
    else:
        raise TypeError("Unknown argument type")

    for key, value in parsed_args.as_dict().items():
        if value is None:
            continue
        args.append(f"--{key}")
        if (
            isinstance(value, list)
            or isinstance(value, tuple)
            or isinstance(value, set)
        ):
            args.append(" ".join([str(item) for item in value]))
        else:
            args.append(str(value))

        if any(
            [
                " " in args[-1],
                "[" in args[-1],
                "]" in args[-1],
                "(" in args[-1],
                ")" in args[-1],
            ]
        ):
            args[-1] = f"'\"{args[-1]}\"'"

    command = r" ".join(args)
    logger.info(command)

    if dry_run:
        return command  # , ""

    result = subprocess.run(command, cwd=dir_path, capture_output=True, text=True)
    return result.stdout  # , result.stderr


if __name__ == "__main__":
    args = TrainArgs().parse_args()
    print(pueue_submit(args, dry_run=True))
    args.run_name = "Name with special [(character)]"
    args.exp_name = "Name with space"
    print(pueue_submit(args, "Non-Default Group", dry_run=True))
    # print(pueue_submit(args, "Non-Default Group"))
    print(pueue_submit(args, "Non-Default Group", 3))
    resume_args = ResumeArgs().parse_args(
        ["--resume_run_id", "59839d3231604cb093053cf76cec5143"]
    )
    print(pueue_submit(resume_args, "ResumeGroup"))
