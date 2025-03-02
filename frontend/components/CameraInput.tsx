"use client"
import { Webcam } from "@webcam/react";
import { Button } from "./ui/button";
import { useState } from "react";
import Image from "next/image";

function CameraInput() {
    const [snapShot, setSnapShot] = useState<string>("")
    return (
        <div>
            <Webcam>
                {({ getSnapshot }) => <Button onClick={() => {
                    console.log("Button Clicked")
                    const pic = getSnapshot({ quality: 0.8 })
                    setSnapShot(pic ?? "")
                    console.log(pic)
                }}>
                    Take Photo
                </Button>}
            </Webcam>
            <div>
                <Image src={snapShot} width={1000} height={1000} alt="Current snapshot" />
            </div>
        </div>

    );
}

export default CameraInput;