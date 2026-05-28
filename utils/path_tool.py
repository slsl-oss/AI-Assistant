"""
为整个工程提供统一的绝对路径
"""
import os.path


def get_project_path() -> str:
    """
    获取项目根目录
    :return: 项目根目录
    """

    #当前文件的绝对路径 os.path.abspath(__file__)
    current_file = os.path.abspath(__file__)

    #获取工程的根目录路径，先获取文件所在的文件夹的绝对路径
    current_dir = os.path.dirname(current_file)

    #获取根目录
    project_root = os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path : str)-> str:
    """
    传递相对路径，获取绝对路径
    :return:
    """
    project_root = get_project_path()
    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))
