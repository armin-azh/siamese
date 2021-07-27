package main

import (
	"./controller/compressor"
	"fmt"
)

func main(){

	var filename = "armin.zip"
	var target = "data/temp"
	var outputPath = "data"
    err:=compressor.Zip(filename,outputPath,target)

    fmt.Println(err)

}