#!/bin/bash
#SBATCH -A info5100.202610 -p sharing,gpu --gres=gpu:h200:1 -t 01:00:00 -c 16 --mem=32G
#SBATCH -o /scratch/%u/info7375-w9/logs/%x-%j.out
# One hour of a Flow-GRPO run:  sbatch -J <run> cluster/train.sbatch --lr 1e-4
# train.py writes a checkpoint after every step and --resume continues from the last one. Each link
# first queues the next link, which waits for this one to end; if train.py returns (run finished or
# crashed) the next link is cancelled, so the chain only continues after a time-limit kill.
# A chain holds two jobs and the sharing QoS allows 4 queued jobs per user (a job that lists sharing
# counts there): two chains at most, and the evaluations stay off sharing.
set -u
source /scratch/$USER/info7375-w9/code/cluster/env.sh
# A requeued link must not run next to the live one: both would write the same checkpoints.
if squeue -h -u $USER -n "$SLURM_JOB_NAME" -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID" | grep -q .; then
  echo "another link of $SLURM_JOB_NAME is running"; exit 0
fi
# Queue the next link in the background, retrying while the QoS limit on queued jobs is reached.
NEXT_FILE=$W/runs/$SLURM_JOB_NAME.next
rm -f $NEXT_FILE
(until sbatch --parsable --dependency=afterany:$SLURM_JOB_ID -J "$SLURM_JOB_NAME" cluster/train.sbatch "$@" \
   > $NEXT_FILE; do echo "$(date) could not queue the next link, retrying in 5 minutes"; sleep 300; done) &
QUEUER=$!
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python train.py --out $W/runs/$SLURM_JOB_NAME --resume "$@"
STATUS=$?
kill $QUEUER 2>/dev/null
NEXT=$(cat $NEXT_FILE 2>/dev/null)
echo "$(date) train.py exited with $STATUS; cancelling the next link $NEXT"
[ -n "$NEXT" ] && scancel $NEXT
