./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB64a4b --dtag=nhwc mb1_ic64oc64_ih56oh58kh3sh1dh2ph1_iw56ow58kw3sw1dw2pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB64a4b --dtag=nhwc mb1_ic64oc64_ih56oh56kh1sh1dh2ph0_iw56ow56kw1sw1dw2pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB64a4b --dtag=nhwc mb1_ic128oc128_ih28oh30kh3sh1dh2ph1_iw28ow30kw3sw1dw2pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB32a4b --dtag=nhwc mb1_ic128oc256_ih28oh14kh1sh2dh2ph0_iw28ow14kw1sw2dw2pw0
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB64a4b --dtag=nhwc mb1_ic256oc256_ih14oh16kh3sh1dh2ph1_iw14ow16kw3sw1dw2pw1
./benchdnn --conv --engine=cpu:0 --verbose=1 --cfg=u8s8s32 --dir=FWD_I --mode=p --fixtimes-per-prb=100 --stag=nhwc --wtag=AcdB48a4b --dtag=nhwc mb1_ic256oc512_ih14oh7kh1sh2dh2ph0_iw14ow7kw1sw2dw2pw0
