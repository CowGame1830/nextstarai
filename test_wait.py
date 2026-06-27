import subprocess
import time

video = r'C:\Users\USER\Desktop\Code\Project\football_analysis\output_videos\22_20260416_202517_output.avi'
print('Opening video with start /wait...')
print(f'Time: {time.time()}')

result = subprocess.run(
    f'start /wait "{video}"',
    shell=True,
    cwd='C:\\Windows\\System32'
)

print(f'Finished at: {time.time()}')
print(f'Return code: {result.returncode}')
