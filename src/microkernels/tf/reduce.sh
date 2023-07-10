
REAPEAT_TIME=100

python3 -u src/reduce.py 0 $REAPEAT_TIME #2>&1 | tee $LOG_DIR/reduce0_128_512_1024.log
#python3 -u $CODE_DIR/reduce.py 1 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/reduce1_65536_1024.log
#python3 -u $CODE_DIR/reduce.py 2 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/reduce2_128_4032_11_11.log
#python3 -u $CODE_DIR/reduce.py 3 $REAPEAT_TIME 2>&1 | tee $LOG_DIR/reduce3_128_2048_7_7.log