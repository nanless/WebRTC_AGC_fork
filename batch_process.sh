mkdir -p ICASSP2024_blind_data_agc_adaptivegain
for x in `ls /data3/zhounan/codes/github_repos/SIG-Challenge/ICASSP2024/blind_data/*.wav`; do
    echo $x
    name=`basename $x`
    ./build/agc $x ICASSP2024_blind_data_agc_adaptivegain/$name
done