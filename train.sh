#!/bin/bash

set -x
set -e

SHORT=n:s:
LONG=name:script:

PARSED=`getopt --options $SHORT --longoptions $LONG --name "$0" -- "$@"`
if [[ $? != 0 ]]; then
  exit 2
fi
eval set -- "$PARSED"

SCRIPT="train"
NAME=""

while true; do
  case "$1" in
    -s|--script)
      SCRIPT="$2"
      shift 2
      ;;
    -n|--name)
      NAME="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid flag"
      exit 3
      ;;
  esac
done

if [[ -z "$NAME" ]]; then
	echo "Name is not specified"
	exit 4
fi

LOG="./logs/$NAME.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

DATA_PATH="./datasets/CelebA"
SNAPSHOTS_PATH="./snapshots/$NAME"

mkdir -p $SNAPSHOTS_PATH
echo -e '\033k'$NAME'\033\\'
unbuffer th ./$SCRIPT.lua --data_path $DATA_PATH --snapshot_path $SNAPSHOTS_PATH $@
