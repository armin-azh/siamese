package logging

import(
	"github.com/fatih/color"
	"log"
)


func Println(level string,msg string){
	switch level {

	case DEBUG:
		color.Set(color.FgBlue)
		defer color.Unset()
		log.Println(msg)
	case DANG:
		color.Set(color.FgRed)
		defer color.Unset()
		log.Println(msg)
	case WARN:
		color.Set(color.FgHiYellow)
		defer color.Unset()
		log.Println(msg)
	case WARNING:
		color.Set(color.FgYellow)
		defer color.Unset()
		log.Println(msg)
	case INFO:
		color.Set(color.FgHiBlue)
		defer color.Unset()
		log.Println(msg)
	case SUCCESS:
		color.Set(color.FgGreen)
		defer color.Unset()
		log.Println(msg)
	default:
		log.Println(msg)
	}
}
