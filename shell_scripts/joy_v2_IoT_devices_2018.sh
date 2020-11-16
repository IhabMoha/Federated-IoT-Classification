ethr_addr_list="d0:52:a8:00:67:5e
                44:65:0d:56:cc:d3
                70:ee:50:18:34:43
                f4:f2:6d:93:51:f1
                00:16:6c:ab:6b:88
                30:8c:fb:2f:e4:b2
                00:62:6e:51:27:2e
                e8:ab:fa:19:de:4f
                00:24:e4:11:18:a8
                ec:1a:59:79:f4:89
                50:c7:bf:00:56:39
                74:c6:3b:29:d7:1d
                ec:1a:59:83:28:11
                18:b4:30:25:be:e4
                70:ee:50:03:b8:ac
                00:24:e4:1b:6f:96
                74:6a:89:00:2e:25
                00:24:e4:20:28:c6
                d0:73:d5:01:83:08
                18:b7:9e:02:20:44
                e0:76:d0:33:bb:85
                70:5a:0f:e4:9b:c0
                08:21:ef:3b:fc:e3
                30:8c:fb:b6:ea:45"

fileCo=21
while [ $fileCo -le 47 ]
do
    fileName="${fileCo}.pcap"
    echo Processing $fileName
    deviceCo=1
    for addr in $ethr_addr_list
    do
        joy output="json_files/${fileCo}_${deviceCo}.json"\
            bpf="ether host ${addr}"\
            bidir=1\
            dns=1\
            http=1\
            "pcap_files/${fileCo}.pcap"
        ((deviceCo++))
    done
    ((fileCo++))
done
echo Done
