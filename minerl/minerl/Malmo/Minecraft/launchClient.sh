#!/bin/bash
port=0; env=0; seed="NONE"; runDir="run"; performanceDir="NONE"; gpu=0
while [ $# -gt 0 ]; do case "$1" in
    -port) port="$2"; shift;; -seed) seed="$2"; shift;;
    -env) env=1;; -runDir) runDir="$2"; shift;;
    -performanceDir) performanceDir="$2"; shift;;
    -gpu) gpu="$2"; shift;;
    -replaceable) ;;
esac; shift; done

mkdir -p "$runDir"/config
cat > "$runDir"/config/malmomodCLIENT.cfg << CFGEOF
malmoports { I:portOverride=$port }
malmoscore { I:policy=0 }
malmoperformance { I:outDir=$performanceDir }
malmoseed { I:seed=$seed }
CFGEOF
[ $env -gt 0 ] && echo "envtype { B:env=true }" >> "$runDir"/config/malmomodCLIENT.cfg

exec docker run --rm -t --user 1000:1000 --gpus "\"device=$gpu\"" \
    -v /data1/gpuadmin/c-act:/app/repo \
    -v /home/gpuadmin/.gradle:/home/gpuadmin/.gradle \
    --tmpfs /app/repo/minerl/minerl/Malmo/Minecraft/$runDir:exec,uid=1000,gid=1000 \
    --shm-size=4g --network host \
    -e HOME=/home/gpuadmin \
    sjlee1218/xenon:latest bash -c "
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &
sleep 2
cd /app/repo/minerl/minerl/Malmo/Minecraft
mkdir -p $runDir/config
exec ./gradlew runClient --no-daemon -PrunDir=$runDir -x getAssets
"
