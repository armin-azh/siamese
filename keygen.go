package main

import (
	"./controller/identifier"
	"./controller/logging"
	"fmt"
	"os"
	"time"
)

func main(){
	args :=os.Args
	var outputPath string
	if len(args)<2{
		outputPath = "data/keys"
	}else if args[1]=="-o"{
		if len(args)<3{
			logging.Println(logging.WARNING,"No value for -o option")
			outputPath="data/keys"
		}
		outputPath=args[2]
	}
	cipher,err:=identifier.CreateNewCipher(identifier.RSA,"")
	if err!=nil{
		fmt.Println(err)
	}
	logging.Println(logging.INFO,"[Create] rsa key...")
	keys,err:=cipher.GenerateKey(outputPath)
	if err!=nil{
		fmt.Println(err)
	}
	logging.Println(logging.SUCCESS,"[Private Key] save at "+keys["Private"])
	logging.Println(logging.SUCCESS,"[Public Key] save at "+keys["Public"])

	time.Sleep(2000*time.Millisecond)
}