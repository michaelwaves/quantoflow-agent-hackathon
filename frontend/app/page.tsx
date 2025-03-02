import CameraInput from "@/components/CameraInput";
import { Toaster } from "@/components/ui/sonner";

export default function Home() {
  return (
    <div className="">
      <div>
        <CameraInput />
      </div>
      <Toaster />
    </div>
  );
}
