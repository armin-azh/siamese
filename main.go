package main

import (
	"./controller/compressor"
)

func main(){

	var filename = "armin.zip"
	var target = "/tmp/armin"
	var outputPath = "./data/temp"
    compressor.Zip(filename,outputPath,target)
}