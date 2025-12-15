# Summary (best-by-accuracy)

| model        | attack   |      eps | defense   | params   |      acc |   acc_adv |   clean_acc |   clean_def_acc |   clean_penalty |   time_ms_per_img | commit                                   | notes               | split   | device   | tag   |     drop |   recovery | run_id             |
|:-------------|:---------|---------:|:----------|:---------|---------:|----------:|------------:|----------------:|----------------:|------------------:|:-----------------------------------------|:--------------------|:--------|:---------|:------|---------:|-----------:|:-------------------|
| ResNet18_STD | fgsm     | 0.015686 | none      | {}       | 0.246094 |  0.246094 |    0.949219 |        0.949219 |               0 |          0.235026 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.703125 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.031373 | none      | {}       | 0.179688 |  0.179688 |    0.949219 |        0.949219 |               0 |          0.234202 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.769531 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.047059 | none      | {}       | 0.140625 |  0.140625 |    0.949219 |        0.949219 |               0 |          0.235934 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.808594 |          0 | 1765792574-65c18f9 |


# Summary (best-by-tradeoff)

| model        | attack   |      eps | defense   | params   |      acc |   acc_adv |   clean_acc |   clean_def_acc |   clean_penalty |   time_ms_per_img | commit                                   | notes               | split   | device   | tag   |     drop |   recovery | run_id             |
|:-------------|:---------|---------:|:----------|:---------|---------:|----------:|------------:|----------------:|----------------:|------------------:|:-----------------------------------------|:--------------------|:--------|:---------|:------|---------:|-----------:|:-------------------|
| ResNet18_STD | fgsm     | 0.015686 | none      | {}       | 0.246094 |  0.246094 |    0.949219 |        0.949219 |               0 |          0.235026 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.703125 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.031373 | none      | {}       | 0.179688 |  0.179688 |    0.949219 |        0.949219 |               0 |          0.234202 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.769531 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.047059 | none      | {}       | 0.140625 |  0.140625 |    0.949219 |        0.949219 |               0 |          0.235934 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.808594 |          0 | 1765792574-65c18f9 |


# Full Results

| model        | attack   |      eps | defense   | params   |      acc |   acc_adv |   clean_acc |   clean_def_acc |   clean_penalty |   time_ms_per_img | commit                                   | notes               | split   | device   | tag   |     drop |   recovery | run_id             |
|:-------------|:---------|---------:|:----------|:---------|---------:|----------:|------------:|----------------:|----------------:|------------------:|:-----------------------------------------|:--------------------|:--------|:---------|:------|---------:|-----------:|:-------------------|
| ResNet18_STD | fgsm     | 0.015686 | none      | {}       | 0.246094 |  0.246094 |    0.949219 |        0.949219 |               0 |          0.235026 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.703125 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.031373 | none      | {}       | 0.179688 |  0.179688 |    0.949219 |        0.949219 |               0 |          0.234202 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.769531 |          0 | 1765792574-65c18f9 |
| ResNet18_STD | fgsm     | 0.047059 | none      | {}       | 0.140625 |  0.140625 |    0.949219 |        0.949219 |               0 |          0.235934 | 65c18f93a70a03e9431fa890ddd9339ff04359ca | baseline-no-defense | test    | mps      | std   | 0.808594 |          0 | 1765792574-65c18f9 |