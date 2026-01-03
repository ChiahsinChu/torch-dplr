set -e

dp train input.json 1>dp_train.stdout 2>dp_train.stderr
dp freeze 1>dp_freeze.stdout 2>dp_freeze.stderr
dp convert-backend frozen_model.pb frozen_model.pth 1>dp_convert.stdout 2>dp_convert.stderr
