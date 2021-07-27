package main

import (
	"./controller"
	"fmt"
	"os"
)

func main(){

	//var filename = "armin.zip"
	//var target = "data/armin.zip"
	//var outputPath = "data/temp"
    //filenames,err:=compressor.Unzip(target,outputPath)
	//controller.GetCurrentPath()
    fmt.Println(controller.GetCurrentPath())
    //fmt.Println(filenames)
    fmt.Println(os.Getpagesize())

}