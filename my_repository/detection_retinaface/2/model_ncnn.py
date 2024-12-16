import numpy as np
import ncnn
import torch
import time

def test_inference(net):
    torch.manual_seed(0)
    # in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    in0 = np.random.rand(1, 3, 640, 640).astype(np.float32)
    out = []

    # with ncnn.Net() as net:
    #     net.load_param("model.ncnn.param")
    #     net.load_model("model.ncnn.bin")

    # with net.create_extractor() as ex:
    st_time = time.time()
    ex = net.create_extractor()
    ex.input("in0", ncnn.Mat(in0.squeeze(0)).clone())
    print(f"----Duration infer: {time.time()-st_time}")

    st_time = time.time()
    _, out0 = ex.extract("out0")
    out.append(np.array(out0)[0])
    _, out1 = ex.extract("out1")
    out.append(np.array(out1)[0])
    _, out2 = ex.extract("out2")
    out.append(np.array(out2)[0])
    print(f"----Duration post: {time.time()-st_time}")

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    net = ncnn.Net()
    net.load_param("./model.ncnn/model.ncnn.param")
    net.load_model("./model.ncnn/model.ncnn.bin")
    for i in range(10):
        st_time = time.time()
        print(test_inference(net))
        # print(f"----Duration: {time.time()-st_time}")
