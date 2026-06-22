import asyncio
import os
import time
from datetime import datetime

import pytz
import requests
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils.logger_handler import logger
from rag.rag_service import RagSummarizeService
from memory.memory_tools import save_memory, search_memory, _get_current_user_id


# ==================== 百度搜索 ====================

@tool(description="使用百度搜索引擎进行联网搜索")
def baidu_web_search(query: str) -> str:
    """使用百度搜索引擎进行联网搜索。"""
    API_KEY = os.getenv("BAIDU_API_KEY")
    if not API_KEY:
        raise ValueError("请先设置 BAIDU_API_KEY 环境变量")
    url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {"messages": [{"content": query}]}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return str(response.json())
    else:
        return f"搜索失败: {response.status_code}, {response.text}"


# ==================== RAG 检索 ====================

@tool(description=(
    "从本地知识库检索相关文档并总结回答。仅当用户问题涉及已存入向量库的内容时调用："
    "（1）用户追问之前生成的报告中的内容；"
    "（2）问题涉及用户上传的参考文档（服装尺码、面料养护、技术文档等）；"
    "（3）问题明确提到'知识库'、'之前报告'、'上传的文档'等关键词。"
    "简单问候、日期天气、常识问答、联网搜索能解决的新问题 → 不调用此工具。"
))
def rag_summarize(query: str) -> str:
    user_id = _get_current_user_id()
    rag_service = RagSummarizeService(user_id=user_id)
    return rag_service.rag_summarize(query)


# ==================== 中间件触发器 ====================

@tool(description="无入参，无返回值，调用后触发中间件，自动为特定场景填充特定的上下文信息")
def fill_context_for_other_prompt():
    logger.info(f"[fill_context_for_other_prompt]已调用")
    return


# ==================== Agent 能力列表 ====================

@tool(description="列出当前系统可用的所有 Agent 及其能力，当用户询问'你能做什么''有哪些功能''支持什么'时调用")
def list_agents() -> str:
    """列出所有可用的 Agent 及其能力"""
    return """
## 当前系统 Agent 列表

| Agent | 能力 |
|-------|------|
| **react_agent** | 通用助手，处理日常对话、日期时间查询、天气查询、联网搜索、知识库检索、记忆管理 |
| **research_agent** | 深度研究引擎，三级并行流水线（子问题→搜索→抓取→压缩→写草稿） |
| **editor_agent** | 子编排器，规划报告大纲，协调 research→review→revise 循环 |
| **reviewer_agent** | 草稿审查，检查信息准确性、完整性、逻辑连贯性 |
| **reviser_agent** | 草稿修改，根据审查意见修订内容 |
| **writer_agent** | 报告撰写，生成引言和结论 |
| **publisher_agent** | 排版导出，生成完整 Markdown 格式报告 |
| **human_agent** | 人工审批，检查大纲和最终报告 |

### 工作机制
- **简单问题**（日期、天气、闲聊、常识）：直接由 react_agent 处理
- **深度研究**（技术调研、行业分析、报告生成）：STORM 流水线
  browser → planner → human → researcher → writer → publisher
- **子编排**：每个章节独立运行 research → review → revise 循环，确保质量
"""


# ==================== 日期工具 ====================

def _get_time_from_multiple_apis(timezone: str) -> tuple:
    """从多个 API 获取时间，任何一个成功就返回"""
    apis = [
        {
            'url': f"http://worldtimeapi.org/api/timezone/{timezone}",
            'parser': lambda data: (
                datetime.fromisoformat(data['datetime'].replace('Z', '+00:00')),
                data.get('timezone', timezone),
                data.get('utc_offset', ''),
            ),
            'timeout': 5,
        },
        {
            'url': f"https://timeapi.io/api/Time/current/zone?timeZone={timezone}",
            'parser': lambda data: (
                datetime.fromisoformat(data.get('dateTime', '').replace('Z', '+00:00')),
                data.get('timeZone', timezone),
                data.get('timeZone', ''),
            ),
            'timeout': 5,
        },
        {
            'url': None,
            'parser': lambda _: (
                datetime.now(pytz.timezone(timezone)),
                timezone,
                datetime.now(pytz.timezone(timezone)).strftime('%z'),
            ),
            'timeout': None,
        },
    ]
    for api in apis:
        try:
            if api['url'] is None:
                return api['parser'](None)
            response = requests.get(api['url'], timeout=api['timeout'])
            if response.status_code == 200:
                return api['parser'](response.json())
        except Exception as e:
            logger.warning(f"API {api.get('url', 'local')} 失败: {str(e)}")
            continue
    raise Exception("所有 API 都无法获取时间，请检查网络连接")


@tool(description="获取指定时区的当前时间（支持多个备用API）")
def get_current_time_by_timezone(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间"""
    try:
        dt, tz_name, utc_offset = _get_time_from_multiple_apis(timezone)
        weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        return f"""
📍 时区：{tz_name}
📅 日期：{dt.strftime('%Y年%m月%d日')}
🕐 时间：{dt.strftime('%H:%M:%S')}
📆 星期：{weekdays[dt.weekday()]}
🌍 UTC偏移：{utc_offset}
        """.strip()
    except Exception as e:
        logger.error(f"获取时间失败: {str(e)}")
        try:
            utc_now = datetime.utcnow()
            dt = utc_now.replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))
            weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            return f"""
📍 时区：{timezone}
📅 日期：{dt.strftime('%Y年%m月%d日')}
🕐 时间：{dt.strftime('%H:%M:%S')}
📆 星期：{weekdays[dt.weekday()]}
            """.strip()
        except:
            return f"无法获取当前时间。错误：{str(e)}"


@tool(description="获取WorldTimeAPI支持的所有时区列表（带缓存）")
def get_timezone_list(region: str = "") -> str:
    """获取所有支持的时区列表"""
    if not hasattr(get_timezone_list, '_cache'):
        get_timezone_list._cache = None
        get_timezone_list._cache_time = 0

    current_time = time.time()
    if get_timezone_list._cache is None or current_time - get_timezone_list._cache_time > 3600:
        try:
            response = requests.get("http://worldtimeapi.org/api/timezone", timeout=10)
            if response.status_code == 200:
                get_timezone_list._cache = response.json()
                get_timezone_list._cache_time = current_time
        except:
            pass

    all_timezones = get_timezone_list._cache or [
        'Asia/Shanghai', 'Asia/Tokyo', 'Asia/Seoul', 'Asia/Singapore',
        'America/New_York', 'America/Los_Angeles', 'America/Chicago',
        'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Moscow',
        'Australia/Sydney',
    ]

    if region:
        filtered = [tz for tz in all_timezones if tz.startswith(region)]
        if filtered:
            result = f"🌍 {region}地区的时区（共{len(filtered)}个）：\n"
            for i, tz in enumerate(sorted(filtered)[:20], 1):
                result += f"{i}. {tz}\n"
            return result
        regions = set(tz.split('/')[0] for tz in all_timezones if '/' in tz)
        return f"未找到 {region} 地区的时区。可用地区：{', '.join(sorted(regions)[:10])}"

    regions = {}
    for tz in all_timezones:
        if '/' in tz:
            regions[tz.split('/')[0]] = regions.get(tz.split('/')[0], 0) + 1
    result = f"🌍 支持的总时区数：{len(all_timezones)}\n\n📊 各地区时区统计：\n"
    for region_name, count in sorted(regions.items()):
        result += f"  • {region_name}: {count}个\n"
    result += "\n💡 常用时区：Asia/Shanghai, America/New_York, Europe/London, Asia/Tokyo"
    return result


@tool(description="根据城市名称获取当地时间")
def get_city_time(city_name: str) -> str:
    """根据城市名称获取当地时间"""
    city_timezone_map = {
        '北京': 'Asia/Shanghai', '上海': 'Asia/Shanghai', '广州': 'Asia/Shanghai',
        '深圳': 'Asia/Shanghai', '香港': 'Asia/Hong_Kong', '台北': 'Asia/Taipei',
        '东京': 'Asia/Tokyo', '首尔': 'Asia/Seoul', '新加坡': 'Asia/Singapore',
        '曼谷': 'Asia/Bangkok', '纽约': 'America/New_York', '洛杉矶': 'America/Los_Angeles',
        '芝加哥': 'America/Chicago', '伦敦': 'Europe/London', '巴黎': 'Europe/Paris',
        '柏林': 'Europe/Berlin', '莫斯科': 'Europe/Moscow',
        '悉尼': 'Australia/Sydney', '迪拜': 'Asia/Dubai', '孟买': 'Asia/Kolkata',
    }
    timezone = city_timezone_map.get(city_name)
    if not timezone:
        for key in city_timezone_map:
            if key in city_name or city_name in key:
                timezone = city_timezone_map[key]
                break
    if not timezone:
        available = list(city_timezone_map.keys())[:15]
        return f"不支持城市 '{city_name}'。支持的城市：{', '.join(available)}"
    return get_current_time_by_timezone(timezone)


@tool(description="比较两个时区或城市的时间差异")
def compare_time(tz1: str, tz2: str) -> str:
    """比较两个时区或城市的时间差异"""
    city_map = {
        '北京': 'Asia/Shanghai', '上海': 'Asia/Shanghai',
        '纽约': 'America/New_York', '伦敦': 'Europe/London',
        '东京': 'Asia/Tokyo', '巴黎': 'Europe/Paris',
        '悉尼': 'Australia/Sydney', '新加坡': 'Asia/Singapore',
    }

    def normalize(name):
        if name in city_map:
            return city_map[name]
        if '/' in name:
            return name
        for city, tz in city_map.items():
            if city in name or name in city:
                return tz
        return name

    timezone1 = normalize(tz1)
    timezone2 = normalize(tz2)

    try:
        dt1, _, offset1 = _get_time_from_multiple_apis(timezone1)
        dt2, _, offset2 = _get_time_from_multiple_apis(timezone2)
        name1 = timezone1.split('/')[-1] if '/' in timezone1 else timezone1
        name2 = timezone2.split('/')[-1] if '/' in timezone2 else timezone2

        def parse_offset(offset_str):
            if not offset_str or offset_str == 'Z' or offset_str == '+00:00':
                return 0
            sign = 1 if offset_str[0] == '+' else -1
            hours = int(offset_str[1:3])
            minutes = int(offset_str[4:6]) if ':' in offset_str and len(offset_str) >= 6 else 0
            return sign * (hours + minutes / 60)

        hour_diff = parse_offset(offset2) - parse_offset(offset1)
        return f"""
📊 时间对比
{'=' * 40}
📍 {name1} ({timezone1})
   🕐 {dt1.strftime('%H:%M:%S')}  📅 {dt1.strftime('%Y-%m-%d')}  🌍 {offset1}
📍 {name2} ({timezone2})
   🕐 {dt2.strftime('%H:%M:%S')}  📅 {dt2.strftime('%Y-%m-%d')}  🌍 {offset2}
{'=' * 40}
⏰ 时差：{name2} 比 {name1} {'早' if hour_diff > 0 else '晚'} {abs(hour_diff):.1f} 小时
        """.strip()
    except Exception as e:
        return f"比较失败：{str(e)}"


# ==================== 天气 MCP 工具 ====================

async def _load_weather_mcp():
    client = MultiServerMCPClient(
        connections={
            "weather": {
                "command": "npx",
                "args": ["@mariox/weather-mcp-server"],
                "transport": "stdio",
            }
        }
    )
    return await client.get_tools()


_weather_mcp_tools = asyncio.run(_load_weather_mcp())
logger.info(f"成功加载天气MCP工具: {[t.name for t in _weather_mcp_tools]}")


# ==================== 工具集合 ====================

base_tools = [
    rag_summarize, fill_context_for_other_prompt, baidu_web_search,
    save_memory, search_memory,
    get_current_time_by_timezone, get_timezone_list, get_city_time, compare_time,
    list_agents,
    *_weather_mcp_tools,
]
logger.info(f"成功加载工具: {[tool.name for tool in base_tools]}")