# 8-AlexNet-tensorflow

## 用1個GPU訓練AlexNet

![AlexNet结构图](https://user-images.githubusercontent.com/55481792/135967068-7f45d4ee-9142-42eb-9e47-b485159916b0.png)

网络结构：

![alexnet](https://user-images.githubusercontent.com/55481792/135819320-01a09e94-1e53-4c0d-8146-03076e9b6c2c.png)


在寫代碼時，注意，optimizer需要初始化。

>init = tf.global_variables_initializer()要写在网络之后。
```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
訓練後 accuracy rate 約爲 51%

預測結果示例：
![predict_results](https://user-images.githubusercontent.com/55481792/135819135-e7442990-283c-4a4a-81a7-847b0a0c0ee2.png)

