#!/bin/bash

# ************************************************** ARGUMENTS ***************************************************
# source: https://stackabuse.com/how-to-parse-command-line-arguments-in-bash/

SHORT=j:,c:,s:,h
LONG=job_name:,cfg:,array_size:,help
OPTS=$(getopt -a -n dataset_creation --options $SHORT --longoption $LONG -- "$@")
eval set -- "$OPTS"
while :
do
  case "$1" in
    -j | --job_name )
      JOB_NAME="$2"
      shift 2
      ;;
    -c | --cfg )
      CFG="$2"
      shift 2
      ;;
    -a | --array_size )
      ARRAY_SIZE="$2"
      shift 2
      ;;
    -h | --help )
      printf "%s\n" "$HELP"
      shift;
      exit 1
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done
echo "Starting job: $JOB_NAME"
echo "Main config file: $CFG"
echo "Job array size: $ARRAY_SIZE"