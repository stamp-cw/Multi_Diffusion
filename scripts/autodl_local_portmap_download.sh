#!/usr/bin/env bash
#ssh 连接的端口号
port="28130"
ssh -CNg -L 8080:127.0.0.1:8080 root@connect.nmb1.seetacloud.com -p ${port}