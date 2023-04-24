import json

stringinput = """Epoch 1/120
284/284 [==============================] - 47s 122ms/step - loss: 3.0254 - accuracy: 0.2699 - val_loss: 2.8712 - val_accuracy: 0.3482 - lr: 0.0100
Epoch 2/120
284/284 [==============================] - 35s 123ms/step - loss: 2.8786 - accuracy: 0.3562 - val_loss: 2.6585 - val_accuracy: 0.4566 - lr: 0.0100
Epoch 3/120
284/284 [==============================] - 33s 118ms/step - loss: 2.7797 - accuracy: 0.4088 - val_loss: 2.7037 - val_accuracy: 0.4328 - lr: 0.0100
Epoch 4/120
284/284 [==============================] - 33s 117ms/step - loss: 2.7026 - accuracy: 0.4422 - val_loss: 2.5940 - val_accuracy: 0.4728 - lr: 0.0100
Epoch 5/120
284/284 [==============================] - 36s 126ms/step - loss: 2.6304 - accuracy: 0.4760 - val_loss: 2.4877 - val_accuracy: 0.5256 - lr: 0.0100
Epoch 6/120
284/284 [==============================] - 35s 123ms/step - loss: 2.5693 - accuracy: 0.5010 - val_loss: 2.4564 - val_accuracy: 0.5357 - lr: 0.0100
Epoch 7/120
284/284 [==============================] - 33s 117ms/step - loss: 2.5183 - accuracy: 0.5226 - val_loss: 2.3626 - val_accuracy: 0.5844 - lr: 0.0100
Epoch 8/120
284/284 [==============================] - 33s 115ms/step - loss: 2.4666 - accuracy: 0.5420 - val_loss: 2.4758 - val_accuracy: 0.5179 - lr: 0.0100
Epoch 9/120
284/284 [==============================] - 36s 125ms/step - loss: 2.4202 - accuracy: 0.5620 - val_loss: 2.3684 - val_accuracy: 0.5498 - lr: 0.0100
Epoch 10/120
284/284 [==============================] - 32s 113ms/step - loss: 2.3854 - accuracy: 0.5705 - val_loss: 2.3665 - val_accuracy: 0.5681 - lr: 0.0100
Epoch 11/120
284/284 [==============================] - 36s 125ms/step - loss: 2.3432 - accuracy: 0.5849 - val_loss: 2.2062 - val_accuracy: 0.6243 - lr: 0.0100
Epoch 12/120
284/284 [==============================] - 33s 116ms/step - loss: 2.3085 - accuracy: 0.5943 - val_loss: 2.2592 - val_accuracy: 0.5887 - lr: 0.0100
Epoch 13/120
284/284 [==============================] - 33s 117ms/step - loss: 2.2710 - accuracy: 0.6079 - val_loss: 2.1964 - val_accuracy: 0.6299 - lr: 0.0100
Epoch 14/120
284/284 [==============================] - 33s 114ms/step - loss: 2.2413 - accuracy: 0.6161 - val_loss: 2.0883 - val_accuracy: 0.6742 - lr: 0.0100
Epoch 15/120
284/284 [==============================] - 33s 117ms/step - loss: 2.2098 - accuracy: 0.6252 - val_loss: 2.3334 - val_accuracy: 0.5641 - lr: 0.0100
Epoch 16/120
284/284 [==============================] - 34s 119ms/step - loss: 2.1797 - accuracy: 0.6343 - val_loss: 2.1982 - val_accuracy: 0.5989 - lr: 0.0100
Epoch 17/120
284/284 [==============================] - 33s 115ms/step - loss: 2.1506 - accuracy: 0.6392 - val_loss: 2.1531 - val_accuracy: 0.6213 - lr: 0.0100
Epoch 18/120
284/284 [==============================] - 34s 119ms/step - loss: 2.1223 - accuracy: 0.6460 - val_loss: 1.9183 - val_accuracy: 0.7281 - lr: 0.0100
Epoch 19/120
284/284 [==============================] - 36s 127ms/step - loss: 2.0944 - accuracy: 0.6546 - val_loss: 2.1778 - val_accuracy: 0.5731 - lr: 0.0100
Epoch 20/120
284/284 [==============================] - 34s 119ms/step - loss: 2.0726 - accuracy: 0.6579 - val_loss: 1.9775 - val_accuracy: 0.6825 - lr: 0.0100
Epoch 21/120
284/284 [==============================] - 36s 127ms/step - loss: 2.0476 - accuracy: 0.6632 - val_loss: 1.8728 - val_accuracy: 0.7262 - lr: 0.0100
Epoch 22/120
284/284 [==============================] - 34s 120ms/step - loss: 2.0242 - accuracy: 0.6674 - val_loss: 1.9379 - val_accuracy: 0.6903 - lr: 0.0100
Epoch 23/120
284/284 [==============================] - 34s 120ms/step - loss: 2.0027 - accuracy: 0.6722 - val_loss: 1.9828 - val_accuracy: 0.6601 - lr: 0.0100
Epoch 24/120
284/284 [==============================] - 36s 127ms/step - loss: 1.9796 - accuracy: 0.6778 - val_loss: 1.8237 - val_accuracy: 0.7305 - lr: 0.0100
Epoch 25/120
284/284 [==============================] - 36s 126ms/step - loss: 1.9584 - accuracy: 0.6826 - val_loss: 1.9293 - val_accuracy: 0.6746 - lr: 0.0100
Epoch 26/120
284/284 [==============================] - 36s 125ms/step - loss: 1.9370 - accuracy: 0.6865 - val_loss: 1.7398 - val_accuracy: 0.7648 - lr: 0.0100
Epoch 27/120
284/284 [==============================] - 36s 126ms/step - loss: 1.9155 - accuracy: 0.6891 - val_loss: 1.8346 - val_accuracy: 0.7092 - lr: 0.0100
Epoch 28/120
284/284 [==============================] - 33s 117ms/step - loss: 1.8977 - accuracy: 0.6939 - val_loss: 1.8847 - val_accuracy: 0.6711 - lr: 0.0100
Epoch 29/120
284/284 [==============================] - 34s 118ms/step - loss: 1.8759 - accuracy: 0.6970 - val_loss: 1.7645 - val_accuracy: 0.7292 - lr: 0.0100
Epoch 30/120
284/284 [==============================] - 34s 119ms/step - loss: 1.8574 - accuracy: 0.6990 - val_loss: 1.7212 - val_accuracy: 0.7442 - lr: 0.0100
Epoch 31/120
284/284 [==============================] - 35s 123ms/step - loss: 1.8399 - accuracy: 0.7015 - val_loss: 1.7197 - val_accuracy: 0.7375 - lr: 0.0100
Epoch 32/120
284/284 [==============================] - 35s 124ms/step - loss: 1.8188 - accuracy: 0.7070 - val_loss: 1.6553 - val_accuracy: 0.7598 - lr: 0.0100
Epoch 33/120
284/284 [==============================] - 33s 116ms/step - loss: 1.8042 - accuracy: 0.7075 - val_loss: 1.6284 - val_accuracy: 0.7714 - lr: 0.0100
Epoch 34/120
284/284 [==============================] - 32s 114ms/step - loss: 1.7857 - accuracy: 0.7108 - val_loss: 1.5929 - val_accuracy: 0.7840 - lr: 0.0100
Epoch 35/120
284/284 [==============================] - 35s 123ms/step - loss: 1.7679 - accuracy: 0.7148 - val_loss: 1.6400 - val_accuracy: 0.7562 - lr: 0.0100
Epoch 36/120
284/284 [==============================] - 33s 116ms/step - loss: 1.7493 - accuracy: 0.7172 - val_loss: 1.6712 - val_accuracy: 0.7317 - lr: 0.0100
Epoch 37/120
284/284 [==============================] - 33s 116ms/step - loss: 1.7341 - accuracy: 0.7178 - val_loss: 1.5764 - val_accuracy: 0.7728 - lr: 0.0100
Epoch 38/120
284/284 [==============================] - 35s 125ms/step - loss: 1.7180 - accuracy: 0.7204 - val_loss: 1.6423 - val_accuracy: 0.7299 - lr: 0.0100
Epoch 39/120
284/284 [==============================] - 33s 114ms/step - loss: 1.6994 - accuracy: 0.7242 - val_loss: 1.6932 - val_accuracy: 0.7035 - lr: 0.0100
Epoch 40/120
1135/1135 [==============================] - 10s 8ms/step
284/284 [==============================] - 48s 168ms/step - loss: 1.6880 - accuracy: 0.7230 - val_loss: 1.6403 - val_accuracy: 0.7238 - lr: 0.0100 - pruning_ratio: 0.1000 - num_examples_pruned: 3630.0000
Epoch 41/120
284/284 [==============================] - 30s 105ms/step - loss: 1.4761 - accuracy: 0.8200 - val_loss: 1.4049 - val_accuracy: 0.8446 - lr: 1.0000e-03
Epoch 42/120
284/284 [==============================] - 30s 107ms/step - loss: 1.4542 - accuracy: 0.8303 - val_loss: 1.3909 - val_accuracy: 0.8522 - lr: 1.0000e-03
Epoch 43/120
284/284 [==============================] - 33s 117ms/step - loss: 1.4464 - accuracy: 0.8334 - val_loss: 1.3982 - val_accuracy: 0.8441 - lr: 1.0000e-03
Epoch 44/120
284/284 [==============================] - 32s 113ms/step - loss: 1.4397 - accuracy: 0.8367 - val_loss: 1.3880 - val_accuracy: 0.8510 - lr: 1.0000e-03
Epoch 45/120
284/284 [==============================] - 32s 114ms/step - loss: 1.4363 - accuracy: 0.8373 - val_loss: 1.3916 - val_accuracy: 0.8486 - lr: 1.0000e-03
Epoch 46/120
284/284 [==============================] - 32s 113ms/step - loss: 1.4310 - accuracy: 0.8398 - val_loss: 1.3902 - val_accuracy: 0.8474 - lr: 1.0000e-03
Epoch 47/120
284/284 [==============================] - 33s 115ms/step - loss: 1.4281 - accuracy: 0.8411 - val_loss: 1.3840 - val_accuracy: 0.8496 - lr: 1.0000e-03
Epoch 48/120
284/284 [==============================] - 33s 115ms/step - loss: 1.4259 - accuracy: 0.8412 - val_loss: 1.3887 - val_accuracy: 0.8474 - lr: 1.0000e-03
Epoch 49/120
284/284 [==============================] - 31s 108ms/step - loss: 1.4244 - accuracy: 0.8418 - val_loss: 1.3902 - val_accuracy: 0.8476 - lr: 1.0000e-03
Epoch 50/120
1022/1022 [==============================] - 9s 9ms/step
284/284 [==============================] - 42s 148ms/step - loss: 1.4168 - accuracy: 0.8444 - val_loss: 1.3923 - val_accuracy: 0.8434 - lr: 1.0000e-03 - pruning_ratio: 0.1000 - num_examples_pruned: 3267.0000
Epoch 51/120
284/284 [==============================] - 30s 104ms/step - loss: 1.2373 - accuracy: 0.9300 - val_loss: 1.3779 - val_accuracy: 0.8500 - lr: 1.0000e-03
Epoch 52/120
284/284 [==============================] - 27s 96ms/step - loss: 1.2320 - accuracy: 0.9309 - val_loss: 1.3861 - val_accuracy: 0.8462 - lr: 1.0000e-03
Epoch 53/120
284/284 [==============================] - 27s 96ms/step - loss: 1.2297 - accuracy: 0.9304 - val_loss: 1.3943 - val_accuracy: 0.8406 - lr: 1.0000e-03
Epoch 54/120
284/284 [==============================] - 30s 104ms/step - loss: 1.2241 - accuracy: 0.9320 - val_loss: 1.3883 - val_accuracy: 0.8459 - lr: 1.0000e-03
Epoch 55/120
284/284 [==============================] - 27s 95ms/step - loss: 1.2226 - accuracy: 0.9322 - val_loss: 1.3733 - val_accuracy: 0.8498 - lr: 1.0000e-03
Epoch 56/120
284/284 [==============================] - 27s 95ms/step - loss: 1.2199 - accuracy: 0.9323 - val_loss: 1.3774 - val_accuracy: 0.8497 - lr: 1.0000e-03
Epoch 57/120
284/284 [==============================] - 30s 104ms/step - loss: 1.2142 - accuracy: 0.9353 - val_loss: 1.3784 - val_accuracy: 0.8478 - lr: 1.0000e-03
Epoch 58/120
284/284 [==============================] - 30s 104ms/step - loss: 1.2129 - accuracy: 0.9352 - val_loss: 1.4118 - val_accuracy: 0.8314 - lr: 1.0000e-03
Epoch 59/120
284/284 [==============================] - 27s 96ms/step - loss: 1.2110 - accuracy: 0.9347 - val_loss: 1.3822 - val_accuracy: 0.8460 - lr: 1.0000e-03
Epoch 60/120
920/920 [==============================] - 8s 8ms/step
284/284 [==============================] - 41s 145ms/step - loss: 1.2079 - accuracy: 0.9367 - val_loss: 1.3711 - val_accuracy: 0.8492 - lr: 1.0000e-03 - pruning_ratio: 0.1000 - num_examples_pruned: 2941.0000
Epoch 61/120
284/284 [==============================] - 26s 92ms/step - loss: 1.1429 - accuracy: 0.9705 - val_loss: 1.3665 - val_accuracy: 0.8527 - lr: 1.0000e-03
Epoch 62/120
284/284 [==============================] - 26s 91ms/step - loss: 1.1396 - accuracy: 0.9720 - val_loss: 1.3673 - val_accuracy: 0.8499 - lr: 1.0000e-03
Epoch 63/120
284/284 [==============================] - 26s 92ms/step - loss: 1.1354 - accuracy: 0.9748 - val_loss: 1.3791 - val_accuracy: 0.8459 - lr: 1.0000e-03
Epoch 64/120
284/284 [==============================] - 26s 90ms/step - loss: 1.1356 - accuracy: 0.9727 - val_loss: 1.3767 - val_accuracy: 0.8445 - lr: 1.0000e-03
Epoch 65/120
284/284 [==============================] - 28s 99ms/step - loss: 1.1308 - accuracy: 0.9747 - val_loss: 1.3760 - val_accuracy: 0.8472 - lr: 1.0000e-03
Epoch 66/120
284/284 [==============================] - 25s 89ms/step - loss: 1.1276 - accuracy: 0.9752 - val_loss: 1.3847 - val_accuracy: 0.8430 - lr: 1.0000e-03
Epoch 67/120
284/284 [==============================] - 27s 96ms/step - loss: 1.1291 - accuracy: 0.9745 - val_loss: 1.3693 - val_accuracy: 0.8489 - lr: 1.0000e-03
Epoch 68/120
284/284 [==============================] - 28s 98ms/step - loss: 1.1263 - accuracy: 0.9750 - val_loss: 1.3758 - val_accuracy: 0.8467 - lr: 1.0000e-03
Epoch 69/120
284/284 [==============================] - 28s 99ms/step - loss: 1.1229 - accuracy: 0.9768 - val_loss: 1.3762 - val_accuracy: 0.8467 - lr: 1.0000e-03
Epoch 70/120
828/828 [==============================] - 7s 8ms/step
284/284 [==============================] - 38s 135ms/step - loss: 1.1241 - accuracy: 0.9757 - val_loss: 1.3712 - val_accuracy: 0.8464 - lr: 1.0000e-03 - pruning_ratio: 0.1000 - num_examples_pruned: 2647.0000
Epoch 71/120
284/284 [==============================] - 26s 93ms/step - loss: 1.1037 - accuracy: 0.9860 - val_loss: 1.3948 - val_accuracy: 0.8361 - lr: 1.0000e-03
Epoch 72/120
284/284 [==============================] - 27s 93ms/step - loss: 1.1033 - accuracy: 0.9849 - val_loss: 1.3822 - val_accuracy: 0.8410 - lr: 1.0000e-03
Epoch 73/120
284/284 [==============================] - 26s 91ms/step - loss: 1.1018 - accuracy: 0.9849 - val_loss: 1.3687 - val_accuracy: 0.8467 - lr: 1.0000e-03
Epoch 74/120
284/284 [==============================] - 24s 83ms/step - loss: 1.0974 - accuracy: 0.9872 - val_loss: 1.3664 - val_accuracy: 0.8499 - lr: 1.0000e-03
Epoch 75/120
284/284 [==============================] - 24s 84ms/step - loss: 1.0961 - accuracy: 0.9873 - val_loss: 1.3838 - val_accuracy: 0.8383 - lr: 1.0000e-03
Epoch 76/120
284/284 [==============================] - 23s 80ms/step - loss: 1.0949 - accuracy: 0.9871 - val_loss: 1.3794 - val_accuracy: 0.8411 - lr: 1.0000e-03
Epoch 77/120
284/284 [==============================] - 26s 91ms/step - loss: 1.0956 - accuracy: 0.9865 - val_loss: 1.3644 - val_accuracy: 0.8470 - lr: 1.0000e-03
Epoch 78/120
284/284 [==============================] - 26s 91ms/step - loss: 1.0951 - accuracy: 0.9866 - val_loss: 1.3648 - val_accuracy: 0.8465 - lr: 1.0000e-03
Epoch 79/120
284/284 [==============================] - 26s 92ms/step - loss: 1.0899 - accuracy: 0.9889 - val_loss: 1.3869 - val_accuracy: 0.8344 - lr: 1.0000e-03
Epoch 80/120
745/745 [==============================] - 6s 8ms/step
284/284 [==============================] - 31s 109ms/step - loss: 1.0899 - accuracy: 0.9877 - val_loss: 1.3719 - val_accuracy: 0.8437 - lr: 1.0000e-03 - pruning_ratio: 0.1000 - num_examples_pruned: 2382.0000
Epoch 81/120
284/284 [==============================] - 22s 76ms/step - loss: 1.0832 - accuracy: 0.9910 - val_loss: 1.3673 - val_accuracy: 0.8458 - lr: 1.0000e-04
Epoch 82/120
284/284 [==============================] - 22s 76ms/step - loss: 1.0815 - accuracy: 0.9913 - val_loss: 1.3675 - val_accuracy: 0.8462 - lr: 1.0000e-04
Epoch 83/120
284/284 [==============================] - 21s 73ms/step - loss: 1.0834 - accuracy: 0.9906 - val_loss: 1.3682 - val_accuracy: 0.8447 - lr: 1.0000e-04
Epoch 84/120
284/284 [==============================] - 21s 73ms/step - loss: 1.0822 - accuracy: 0.9914 - val_loss: 1.3694 - val_accuracy: 0.8437 - lr: 1.0000e-04
Epoch 85/120
284/284 [==============================] - 23s 82ms/step - loss: 1.0813 - accuracy: 0.9924 - val_loss: 1.3693 - val_accuracy: 0.8435 - lr: 1.0000e-04
Epoch 86/120
284/284 [==============================] - 23s 80ms/step - loss: 1.0809 - accuracy: 0.9920 - val_loss: 1.3704 - val_accuracy: 0.8417 - lr: 1.0000e-04
Epoch 87/120
284/284 [==============================] - 21s 75ms/step - loss: 1.0803 - accuracy: 0.9927 - val_loss: 1.3699 - val_accuracy: 0.8429 - lr: 1.0000e-04
Epoch 88/120
284/284 [==============================] - 21s 73ms/step - loss: 1.0814 - accuracy: 0.9916 - val_loss: 1.3695 - val_accuracy: 0.8446 - lr: 1.0000e-04
Epoch 89/120
284/284 [==============================] - 21s 73ms/step - loss: 1.0800 - accuracy: 0.9928 - val_loss: 1.3705 - val_accuracy: 0.8421 - lr: 1.0000e-04
Epoch 90/120
671/671 [==============================] - 5s 8ms/step
284/284 [==============================] - 32s 113ms/step - loss: 1.0799 - accuracy: 0.9924 - val_loss: 1.3728 - val_accuracy: 0.8416 - lr: 1.0000e-04 - pruning_ratio: 0.1000 - num_examples_pruned: 2144.0000
Epoch 91/120
284/284 [==============================] - 22s 78ms/step - loss: 1.0763 - accuracy: 0.9942 - val_loss: 1.3713 - val_accuracy: 0.8428 - lr: 1.0000e-04
Epoch 92/120
116/284 [===========>..................] - ETA: 17s - loss: 1.0753 - accuracy: 0.9948"""

description = input("Enter a description for the data: ")

valacc = []
valloss = []

for lineindex, line in enumerate(stringinput.split("\n")):
  if line.startswith("284/284"):
    acctext = float(line.split("val_accuracy: ")[1].split(" -")[0])
    valacc.append(float(acctext))
    losstext = float(line.split("val_loss: ")[1].split(" -")[0])
    valloss.append(float(losstext))

with open("graphout.json", "r") as file:
  dictionary = json.load(file)
  dictionary[description] = [valacc, valloss]

with open("graphout.json", "w") as file:
  json.dump(dictionary, file)