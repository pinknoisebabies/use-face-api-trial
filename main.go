package main

import (
	"encoding/json"
	"fmt"
	"github.com/lazywei/go-opencv/opencv"
	"io/ioutil"
	"net/http"
	"os"
	"path"
)

func main() {
	win := opencv.NewWindow("Go-OpenCV Webcam Face Detection")
	defer win.Destroy()

	cap := opencv.NewCameraCapture(0)
	if cap == nil {
		panic("cannot open camera")
	}
	defer cap.Release()

	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	cascade := opencv.LoadHaarClassifierCascade(path.Join(cwd, "haarcascade_frontalface_alt.xml"))

	fmt.Println("Press ESC to quit")
	for {
		key := opencv.WaitKey(1)

		if cap.GrabFrame() {
			img := cap.RetrieveFrame(1)
			if img != nil {
				faces := cascade.DetectObjects(img)
				for _, value := range faces {
					opencv.Circle(img,
						opencv.Point{
							value.X() + (value.Width() / 2),
							value.Y() + (value.Height() / 2),
						},
						value.Width()/2,
						opencv.ScalarAll(255.0), 1, 1, 0)
				}

				win.ShowImage(img)

				if key == 32 {
					opencv.SaveImage("test.png", img, nil)
					go post()
				}
			} else {
				fmt.Println("nil image")
			}
		}

		if key == 27 {
			os.Exit(0)
		}
	}
}

func post() {
	file, err := os.Open("test.png")
	defer file.Close()

	c := http.Client{}
	r, err := http.NewRequest("POST", "https://westus.api.cognitive.microsoft.com/face/v1.0/detect?returnFaceAttributes=age", file)
	// You should never ignore the error returned by a call.
	if err != nil {
		panic(err)
	}

	r.Header.Add("Content-Type", "application/octet-stream")
	r.Header.Add("Ocp-Apim-Subscription-Key", os.Getenv("FACEAPIKEY"))

	resp, err := c.Do(r)

	if err != nil {
		fmt.Println(err)
		return
	}

	defer resp.Body.Close()

	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(string(b))
	var data []interface{}
	if err := json.Unmarshal(b, &data); err != nil {
		fmt.Println("JSON Unmarshal error:", err)
		return
	}
	for _, v := range data {
		//faceId := v.(map[string]interface{})["faceId"]
		//faceRectangle := v.(map[string]interface{})["faceRectangle"]
		faceAttributes := v.(map[string]interface{})["faceAttributes"].(map[string]interface{})

		fmt.Printf("%+v", faceAttributes)
	}

}
