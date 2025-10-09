a=0
while true; do
  b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
  [ "$b" -gt "$a" ] && a="$b" && echo "Current Peak: $a MiB"
  sleep 0.5
done