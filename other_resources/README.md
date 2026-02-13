這個資料夾主要用來存一些額外用途的檔案:
sample_manufacture: 用來製造金屬板sample的資料夾
    sample_processing.py: 用來製作金屬sample的所有前置作業檔案包括:放大、連起中央黑色部分、拼湊成一大片等
    convert_PNG2DXF.bat: 將sample_processing.py處理完的PNG檔案轉換成bat的執行檔
    PNG_connectedBlack_senior: 產生將黑色部分連起的PNG data(學長原程式碼)
bad_points.txt: fix_dead_pixels.py所需要的文字檔
BadPixelRepair.py: 把拍攝圖片中的dead pixel修掉(通常是經過imageAverage.py處理過的圖片)
fix_dead_pixels.py: 用途跟BadPixelRepair.py類似。學長寫的
GerchbergSaxtonAlgorith.py: 用多張只有振幅的圖片找出，圖片的真實相位
imageAverage.py: 把拍攝完的圖片多張平均處理雜訊
noise_adder_senior.py: 加入noise到data裡(學長原程式碼)
ONN_weightExtractor.py: 用來讀出ONN的weight(也就是Phase)
ONN_modelVerification.py: 用來測試ONN的檔案1
ONN_modelVerification2.py: 用來測試ONN的檔案2
phasorIntegration.py: 用來計算phasor integration的誤差比例
raw2bmp.py: 將相機拍攝的raw檔轉成bmp可視











