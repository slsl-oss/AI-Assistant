from utils.config_handler import prompts_conf,get_abs_path
from utils.logger_handler import logger


def load_system_prompt():
    """
    加载主提示词，这个是加载到react_agent,当react是调度agent的时候在langchain框架下把其他子agent封装成tool的情况
    :return:
    """
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt]在yaml中没有配置main_prompt_path")
        raise e

    try:
        return open(system_prompt_path,"r",encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompt]解析提示词出错，{str(e)}")
        raise e


def load_rag_prompt():
    """
    加载rag总结agent的提示词
    :return:
    """
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]在yaml中没有配置rag_summarize_prompt_path")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析提示词出错，{str(e)}")
        raise e


def load_date_prompt():
    """
    加载日期agent的提示词
    :return:
    """
    try:
        date_prompt_path = get_abs_path(prompts_conf["date_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_date_prompt]在yaml中没有配置date_prompt_path")
        raise e

    try:
        return open(date_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_date_prompt]解析提示词出错，{str(e)}")
        raise e

def load_weather_prompt():
    """
    加载天气agent的提示词
    :return:
    """
    try:
        weather_prompt_path = get_abs_path(prompts_conf["weather_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_weather_prompt]在yaml中没有配置weather_prompt_path")
        raise e

    try:
        return open(weather_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_weather_prompt]解析提示词出错，{str(e)}")
        raise e

def load_supervisor_prompt():
    """
    加载在langgraph框架下调度agent的提示词，根据用户问题做决策，将问题转发给适合的子agent解决
    :return:
    """
    try:
        supervisor_prompt_path = get_abs_path(prompts_conf["supervisor_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_supervisor_prompt]在yaml中没有配置supervisor_prompt_path")
        raise e

    try:
        return open(supervisor_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_supervisor_prompt]解析提示词出错，{str(e)}")
        raise e

def load_react_prompt():
    """
    加载react_agent作为子agent的时候的提示词
    :return:
    """
    try:
        react_prompt_path = get_abs_path(prompts_conf["react_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_react_prompt]在yaml中没有配置react_prompt_path")
        raise e

    try:
        return open(react_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_react_prompt]解析提示词出错，{str(e)}")
        raise e

if __name__ == '__main__':
    print(load_react_prompt())
