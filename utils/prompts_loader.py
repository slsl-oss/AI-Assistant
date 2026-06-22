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

def load_research_prompt():
    """
    加载research_agent的提示词
    :return:
    """
    try:
        research_prompt_path = get_abs_path(prompts_conf["research_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_research_prompt]在yaml中没有配置research_prompt_path")
        raise e

    try:
        return open(research_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_research_prompt]解析提示词出错，{str(e)}")
        raise e

def load_editor_prompt():
    try:
        path = get_abs_path(prompts_conf["editor_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_editor_prompt]在yaml中没有配置editor_prompt_path")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_editor_prompt]解析提示词出错，{str(e)}")
        raise e

def load_writer_prompt():
    try:
        path = get_abs_path(prompts_conf["writer_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_writer_prompt]在yaml中没有配置writer_prompt_path")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_writer_prompt]解析提示词出错，{str(e)}")
        raise e

def load_reviewer_prompt():
    try:
        path = get_abs_path(prompts_conf["reviewer_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_reviewer_prompt]在yaml中没有配置reviewer_prompt_path")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_reviewer_prompt]解析提示词出错，{str(e)}")
        raise e

def load_reviser_prompt():
    try:
        path = get_abs_path(prompts_conf["reviser_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_reviser_prompt]在yaml中没有配置reviser_prompt_path")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_reviser_prompt]解析提示词出错，{str(e)}")
        raise e

def load_publisher_prompt():
    try:
        path = get_abs_path(prompts_conf["publisher_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_publisher_prompt]在yaml中没有配置publisher_prompt_path")
        raise e
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_publisher_prompt]解析提示词出错，{str(e)}")
        raise e

if __name__ == '__main__':
    print(load_react_prompt())
