import numpy as np
import math
import cv2

interpolation = {
    # Nearest-neighbor interpolation
    1: cv2.INTER_NEAREST,
    # Bilinear interpolation
    2: cv2.INTER_LINEAR,
    # Area interpolation
    3: cv2.INTER_AREA
}

class Proccessing:

    @staticmethod
    def antiShift(data):
        out_data = []
        avg = sum(data) / len(data)
        for i in range(len(data)):
            out_data.append(data[i] - avg)
        return out_data

    @staticmethod
    def antiSpike(data):
        out_data = data
        for i in range(1, len(out_data)-1):
            if math.fabs(out_data[i]*5) > math.fabs(out_data[i+1]):
                out_data[i] = (out_data[i-1]+out_data[i+1])/2
        return out_data

    @staticmethod
    def antiTrendLinear(data, N):
        out_data = []
        for i in range(N - 1):
            out_data.append(data[i + 1] - data[i])
        out_data.append(out_data[N - 2])
        return out_data

    @staticmethod
    def antiTrendNonLinear(data, N, W):
        out_data = []
        for i in range(N - W):
            x_n = 0
            for k in range(W):
                x_n += data[i + k]
            x_n = x_n / W
            out_data.append(x_n)
        # out_data.append(out_data[N - W - 2])
        return out_data

    @staticmethod
    def antiNoise(data, N, M):
        out_data = []
        for i in range(N):
            avg = 0
            for j in range(M):
                avg += data[j]
            avg = avg / M
            out_data.append(avg)
        return out_data

    # @staticmethod
    # def arith_mean(f, buffer_size=10):
    #     buffer = [f] * buffer_size
    #
    #     # Move buffer to actually values ( [0, 1, 2, 3] -> [1, 2, 3, 4] )
    #     buffer = buffer[1:]
    #     buffer.append(f)
    #
    #     # Calculation arithmetic mean
    #     mean = 0
    #     for e in buffer: mean += e
    #     mean /= len(buffer)
    #
    #     return mean

    # @staticmethod
    # def antiNoise(data, N, M):
    #     return [np.mean(data[m][i] for m in range(M)) for i in range(N)]

    @staticmethod
    def lpf(fc, m, dt):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        fact = fc * dt
        lpw = []
        lpw.append(fact)
        arg = fact * math.pi
        for i in range(1, m + 1):
            lpw.append(np.sin(arg * i) / (math.pi * i))
        lpw[m] = lpw[m] / 2
        sumg = lpw[0]
        for i in range(1, m + 1):
            sum = d[0]
            arg = math.pi * i / m
            for k in range(1, 4):
                sum += 2 * d[k] * np.cos(arg * k)
            lpw[i] = lpw[i] * sum
            sumg += 2 * lpw[i]
        for i in range(m + 1):
            lpw[i] = lpw[i] / sumg
        return lpw

    @staticmethod
    def lpf_reverse(lpw):
        return lpw[:0:-1] + lpw

    @staticmethod
    def hpf(fc, m, dt):
        lpw = Proccessing.lpf_reverse(Proccessing.lpf(fc, m, dt))
        hpw = []
        Loper = 2 * m + 1
        for k in range(Loper):
            if k == m:
                hpw.append(1 - lpw[k])
            else:
                hpw.append(- lpw[k])
        return hpw

    @staticmethod
    def bpf(fc1, fc2, m, dt):
        lpw1 = Proccessing.lpf_reverse(Proccessing.lpf(fc1, m, dt))
        lpw2 = Proccessing.lpf_reverse(Proccessing.lpf(fc2, m, dt))
        bpw = []
        Loper = 2 * m + 1
        for k in range(Loper):
            bpw.append(lpw2[k] - lpw1[k])
        return bpw

    @staticmethod
    def bsf(fc1, fc2, m, dt):
        bsw = []
        lpw1 = Proccessing.lpf_reverse(Proccessing.lpf(fc1, m, dt))
        lpw2 = Proccessing.lpf_reverse(Proccessing.lpf(fc2, m, dt))
        Loper = 2 * m + 1
        for k in range(0, Loper):
            if k == m:
                bsw.append(1. + lpw1[k] - lpw2[k])
            else:
                bsw.append(lpw1[k] - lpw2[k])
        return bsw

    @staticmethod
    def frequencyResponse(data, N):
        out_data = []
        for i in range(N):
            out_data.append(data[i] * N)
        return out_data

    @staticmethod
    def shift_2D(img, C):
        img += C
        return img

    @staticmethod
    def multModel_2D(img, C):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = np.uint8(img[i][j] * C)
        return img

    @staticmethod
    def resize(img, scale, typeInt):
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation[typeInt])

    @staticmethod
    def negative(img):
        m = max(img.ravel())
        if img.ndim == 1:
            img = np.reshape(img, (256, 256))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = m - img[i][j]
        return img

    @staticmethod
    def gamma_transform(img, C, gamma):
        if img.ndim == 1:
            img = np.reshape(img, (1024, 1024))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = C * pow(img[i][j], gamma)
        return img

    @staticmethod
    def log_transform(img, C):
        if img.ndim == 1:
            img = np.reshape(img, (1024, 1024))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j] = C * math.log(img[i][j] + 1)
        return img

    @staticmethod
    def gradTransform(data):
        # y = 1. * np.arange(len(data)) / (len(data) - 1)
        data2 = data.copy()
        m = int(data.max())

        count, bins_count = np.histogram(data, bins=max(data)+1, range=[0,256], density=True)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        for i in range(len(data)):
            data2[i] = m * cdf[data[i]]
        return data2

    @staticmethod
    def average_filter(image, kernel_size_x=3, kernel_size_y=3):
        blurred_image = cv2.blur(image, (kernel_size_x, kernel_size_y))
        return blurred_image

    @staticmethod
    def median_filter(image, kernel_size=3):
        filtered_image = cv2.medianBlur(image, kernel_size)
        return filtered_image

    @staticmethod
    def clear(img, threshold):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row, col] <= threshold:
                    img[row, col] = 0
        return img

    @staticmethod
    def improve(img, kernel, alpha=2.5, beta=-0.5):
        filtered = cv2.filter2D(img.copy(), -1, kernel)
        filtered = Proccessing.clear(filtered, 135)
        img_improve = cv2.addWeighted(
            img.copy(), alpha, filtered, beta, 0
        )
        min_val = np.min(img_improve)
        max_val = np.max(img_improve)
        return (img_improve - min_val) / (max_val - min_val), filtered