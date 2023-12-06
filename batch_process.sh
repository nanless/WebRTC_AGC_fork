mkdir -p ICASSP2024_blind_data_agc
for x in `ls ICASSP2024_blind_data/*.wav`; do
    echo $x
    name=`basename $x`
    ./build/agc $x ICASSP2024_blind_data_agc/$name
done