from utils.config_handler import prompts_conf,get_abs_path
from utils.logger_handler import logger


def load_system_prompt():
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

if __name__ == '__main__':
    print(load_supervisor_prompt())
