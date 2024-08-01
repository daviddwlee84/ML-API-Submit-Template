from typing import Union, Optional
import subprocess
import os
import sys
from cli import ResumeArgs
from train import TrainArgs
from loguru import logger
import json

curr_dir = os.path.dirname(os.path.abspath(__file__))


def pueue_set_parallel(
    pueue_group: Optional[str] = None, pueue_parallel: Optional[int] = 1
) -> None:
    if pueue_group:
        # Don't check this since if a group exist it will return 1
        temp_return = subprocess.run(
            ["pueue", "group", "add", pueue_group], capture_output=True
        )
        logger.info(
            f"Create pueue group {pueue_group}: {temp_return.stdout.decode().strip()}"
        )

    if pueue_parallel > 1:
        if pueue_group:
            temp_return = subprocess.run(
                ["pueue", "parallel", "-g", pueue_group, f"{pueue_parallel}"],
                capture_output=True,
                check=True,
            )
            logger.info(
                f"Set parallel for {pueue_group}: {temp_return.stdout.decode().strip()}"
            )
        else:
            temp_return = subprocess.run(
                ["pueue", "parallel", f"{pueue_parallel}"],
                capture_output=True,
                check=True,
            )
            logger.info(f"Set parallel: {temp_return.stdout.decode().strip()}")


def pueue_submit(
    parsed_args: Union[TrainArgs, ResumeArgs],
    pueue_group: Optional[str] = None,
    pueue_parallel: Optional[int] = 1,
    dir_path: str = curr_dir,
    pueue_return_task_id_only: bool = False,
    dry_run: bool = False,
) -> str:  # Tuple[str, str]:
    args = ["pueue", "add"]

    if pueue_group:
        args.extend(["-g", pueue_group])

    if pueue_return_task_id_only:
        # https://github.com/Nukesor/pueue/blob/main/CHANGELOG.md#added-14
        args.append("--print-task-id")

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

        # Special case for boolean flag
        if parsed_args._annotations[key] is bool:
            if value is True:
                args.append(f"--{key}")
            continue

        args.append(f"--{key}")
        if (
            isinstance(value, list)
            or isinstance(value, tuple)
            or isinstance(value, set)
        ):
            # args.append(" ".join([str(item) for item in value]))
            args.extend([str(item) for item in value])
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
            # Since we pass args instead of command string, we can get rid of ""
            args[-1] = f"'{args[-1]}'"

    command = r" ".join(args)
    logger.info(command)

    if dry_run:
        return command  # , ""

    # https://docs.python.org/3/library/subprocess.html#subprocess.run
    result = subprocess.run(
        args, cwd=dir_path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()  # , result.stderr


def pueue_status(task_id: Optional[str] = None) -> dict:
    all_status = json.loads(
        subprocess.run(
            ["pueue", "status", "--json"], stdout=subprocess.PIPE, check=True
        ).stdout.decode()
    )
    if task_id:
        return all_status["tasks"][task_id]
    return all_status


def pueue_logs(task_id: Optional[str] = None) -> dict:
    if task_id:
        return json.loads(
            subprocess.run(
                ["pueue", "log", task_id, "--json"], stdout=subprocess.PIPE, check=True
            ).stdout.decode()
        )[task_id]

    return json.loads(
        subprocess.run(
            ["pueue", "log", "--json"], stdout=subprocess.PIPE, check=True
        ).stdout.decode()
    )


if __name__ == "__main__":
    args: TrainArgs = TrainArgs().parse_args()
    print(pueue_submit(args, dry_run=True))
    args.run_name = "Name with special [(character)]"
    args.exp_name = "Name with space"
    print(pueue_submit(args, "Non-Default Group", dry_run=True))
    args.save_every_epoch = True
    args.save_model = False
    print(pueue_submit(args, "Non-Default Group", dry_run=True))
    exit()
    # print(pueue_submit(args, "Non-Default Group"))
    print(pueue_status())
    print(pueue_logs())
    print(
        task_id := pueue_submit(
            args, "Non-Default Group", 3, pueue_return_task_id_only=True
        )
    )
    print(task_status := pueue_status(task_id))
    print(task_log := pueue_logs(task_id))
    resume_args = ResumeArgs().parse_args(
        ["--resume_run_id", "59839d3231604cb093053cf76cec5143"]
    )
    print(pueue_submit(resume_args, "ResumeGroup"))
