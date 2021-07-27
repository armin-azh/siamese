package main

import (
	"./controller/compressor"
	"fmt"
)

func main(){

	//var filename = "armin.zip"
	var target = "data/armin.zip"
	var outputPath = "data/temp"
    filenames,err:=compressor.Unzip(target,outputPath)

    fmt.Println(err)
    fmt.Println(filenames)

}