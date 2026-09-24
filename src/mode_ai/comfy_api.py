"""
ComfyUI API Client - Gửi workflow tới ComfyUI server
"""
import json
import time
import uuid
import random
from typing import Dict, Optional, Any, List
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


class ComfyUIClient:
    """Client để tương tác với ComfyUI API"""
    
    def __init__(self, server_address: str = "127.0.0.1:8188", client_id: Optional[str] = None):
        self.server_address = server_address
        self.client_id = client_id or str(uuid.uuid4())
        self.base_url = f"http://{server_address}"
    
    def _check_dependencies(self):
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")
    
    def is_server_running(self) -> bool:
        """Kiểm tra ComfyUI server có chạy không"""
        self._check_dependencies()
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def get_system_stats(self) -> Dict:
        """Lấy system stats"""
        self._check_dependencies()
        resp = requests.get(f"{self.base_url}/system_stats", timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def queue_prompt(self, workflow: Dict, prompt_id: Optional[str] = None) -> Dict:
        """Queue workflow vào ComfyUI"""
        self._check_dependencies()
        p = {"prompt": workflow, "client_id": self.client_id}
        if prompt_id:
            p["prompt_id"] = prompt_id
        
        data = json.dumps(p).encode('utf-8')
        resp = requests.post(f"{self.base_url}/prompt", data=data, headers={'Content-Type': 'application/json'})
        resp.raise_for_status()
        return resp.json()
    
    def get_queue(self) -> Dict:
        """Lấy queue status"""
        self._check_dependencies()
        resp = requests.get(f"{self.base_url}/queue")
        resp.raise_for_status()
        return resp.json()
    
    def get_history(self, prompt_id: str) -> Dict:
        """Lấy history của prompt"""
        self._check_dependencies()
        resp = requests.get(f"{self.base_url}/history/{prompt_id}")
        resp.raise_for_status()
        return resp.json()
    
    def get_images(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Lấy image từ ComfyUI output"""
        self._check_dependencies()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        resp = requests.get(f"{self.base_url}/view", params=params)
        resp.raise_for_status()
        return resp.content
    
    def upload_image(self, image_path: str, overwrite: bool = True) -> Dict:
        """Upload image lên ComfyUI input"""
        self._check_dependencies()
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'overwrite': str(overwrite).lower()}
            resp = requests.post(f"{self.base_url}/upload/image", files=files, data=data)
            resp.raise_for_status()
            return resp.json()
    
    def generate_image(self, workflow: Dict, wait: bool = True, timeout: int = 300) -> Optional[Dict]:
        """Generate image và chờ kết quả"""
        self._check_dependencies()
        
        # Queue
        result = self.queue_prompt(workflow)
        prompt_id = result.get('prompt_id')
        print(f"✅ Queued prompt: {prompt_id}")
        
        if not wait:
            return result
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    status = history[prompt_id].get('status', {})
                    if status.get('completed', False):
                        print(f"✅ Generation completed: {prompt_id}")
                        return history[prompt_id]
                    # Check if failed
                    if 'failed' in str(status).lower():
                        print(f"❌ Generation failed: {status}")
                        return history[prompt_id]
            except Exception as e:
                print(f"⚠️ Error checking history: {e}")
            
            time.sleep(2)
        
        print(f"⏱️ Timeout after {timeout}s")
        return None
    
    def list_models(self) -> Dict:
        """List available models"""
        self._check_dependencies()
        try:
            resp = requests.get(f"{self.base_url}/object_info", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except:
            return {}
    
    def interrupt(self):
        """Interrupt current generation"""
        self._check_dependencies()
        data = json.dumps({"client_id": self.client_id}).encode('utf-8')
        resp = requests.post(f"{self.base_url}/interrupt", data=data)
        return resp.json() if resp.content else {}


class MockComfyUIClient:
    """Mock client cho testing khi không có ComfyUI server"""
    
    def __init__(self, *args, **kwargs):
        print("⚠️ Using MockComfyUIClient - no real ComfyUI server")
        self.server_address = "mock"
    
    def is_server_running(self) -> bool:
        return False
    
    def queue_prompt(self, workflow: Dict, prompt_id: Optional[str] = None) -> Dict:
        mock_id = prompt_id or str(uuid.uuid4())
        print(f"🎭 Mock queue: {mock_id}")
        print(f"   Workflow nodes: {len(workflow)}")
        # Save mock workflow for inspection
        Path("output").mkdir(exist_ok=True)
        with open(f"output/mock_workflow_{mock_id[:8]}.json", 'w') as f:
            json.dump(workflow, f, indent=2)
        return {"prompt_id": mock_id, "number": random.randint(1, 100)}
    
    def generate_image(self, workflow: Dict, wait: bool = True, timeout: int = 300) -> Optional[Dict]:
        result = self.queue_prompt(workflow)
        print("🎭 Mock generation - workflow saved to output/")
        return {
            "prompt_id": result["prompt_id"],
            "status": {"completed": True, "mock": True},
            "outputs": {}
        }


def get_client(server_address: str = "127.0.0.1:8188", allow_mock: bool = True) -> ComfyUIClient:
    """Get client, fallback to mock if server not available and allow_mock=True"""
    client = ComfyUIClient(server_address)
    if client.is_server_running():
        print(f"✅ ComfyUI server running at {server_address}")
        return client
    else:
        print(f"⚠️ ComfyUI server not running at {server_address}")
        if allow_mock:
            return MockComfyUIClient()
        else:
            raise ConnectionError(f"ComfyUI server not running at {server_address}")
