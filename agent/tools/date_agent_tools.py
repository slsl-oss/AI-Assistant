import requests
from datetime import datetime
from langchain.tools import tool
from utils.logger_handler import logger
import pytz
import time


# ==================== 多 API 备用方案 ====================

def get_time_from_multiple_apis(timezone: str) -> tuple:
    """
    从多个 API 获取时间，任何一个成功就返回

    Returns:
        (datetime对象, 时区名称, UTC偏移量)
    """
    apis = [
        # WorldTimeAPI
        {
            'url': f"http://worldtimeapi.org/api/timezone/{timezone}",
            'parser': lambda data: (
                datetime.fromisoformat(data['datetime'].replace('Z', '+00:00')),
                data.get('timezone', timezone),
                data.get('utc_offset', '')
            ),
            'timeout': 5
        },
        # TimeAPI (备用)
        {
            'url': f"https://timeapi.io/api/Time/current/zone?timeZone={timezone}",
            'parser': lambda data: (
                datetime.fromisoformat(data.get('dateTime', '').replace('Z', '+00:00')),
                data.get('timeZone', timezone),
                data.get('timeZone', '')
            ),
            'timeout': 5
        },
        # 本地 pytz 计算（最后备选）
        {
            'url': None,
            'parser': lambda _: (
                datetime.now(pytz.timezone(timezone)),
                timezone,
                datetime.now(pytz.timezone(timezone)).strftime('%z')
            ),
            'timeout': None
        }
    ]

    for api in apis:
        try:
            if api['url'] is None:
                # 使用本地 pytz
                return api['parser'](None)

            response = requests.get(api['url'], timeout=api['timeout'])
            if response.status_code == 200:
                return api['parser'](response.json())
            elif response.status_code == 404:
                continue  # 时区不存在，尝试下一个 API
        except Exception as e:
            logger.warning(f"API {api.get('url', 'local')} 失败: {str(e)}")
            continue

    raise Exception("所有 API 都无法获取时间，请检查网络连接")


@tool(description="获取指定时区的当前时间（支持多个备用API）")
def get_current_time_by_timezone(timezone: str = "Asia/Shanghai") -> str:
    """
    获取指定时区的当前时间，使用多个备用API确保稳定性

    Args:
        timezone: 时区名称，如 Asia/Shanghai, America/New_York, Europe/London
    """
    try:
        dt, tz_name, utc_offset = get_time_from_multiple_apis(timezone)

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
        # 最终降级：使用本地时间 + 手动时区偏移
        try:
            # 手动处理常用时区
            timezone_offsets = {
                'Asia/Shanghai': 8,
                'Asia/Tokyo': 9,
                'America/New_York': -5,
                'Europe/London': 0,
                'Europe/Paris': 1,
            }
            utc_now = datetime.utcnow()
            offset_hours = timezone_offsets.get(timezone, 8)
            dt = utc_now.replace(tzinfo=pytz.UTC).astimezone(pytz.timezone(timezone))

            weekdays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
            return f"""
📍 时区：{timezone}
📅 日期：{dt.strftime('%Y年%m月%d日')}
🕐 时间：{dt.strftime('%H:%M:%S')}
📆 星期：{weekdays[dt.weekday()]}
🌍 UTC偏移：UTC{'+' if offset_hours >= 0 else ''}{offset_hours}:00
⚠️ 提示：使用本地计算模式，时间可能略有误差
            """.strip()
        except:
            return f"无法获取当前时间，请检查网络连接和 pytz 安装。错误：{str(e)}"


@tool(description="获取WorldTimeAPI支持的所有时区列表（带缓存）")
def get_timezone_list(region: str = "") -> str:
    """
    获取所有支持的时区列表，带缓存避免重复请求
    """
    # 使用类变量缓存时区列表
    if not hasattr(get_timezone_list, '_cache'):
        get_timezone_list._cache = None
        get_timezone_list._cache_time = 0

    # 缓存1小时
    current_time = time.time()
    if get_timezone_list._cache is None or current_time - get_timezone_list._cache_time > 3600:
        try:
            # 尝试多个API获取时区列表
            urls = [
                "http://worldtimeapi.org/api/timezone",
                "https://raw.githubusercontent.com/eggert/tz/main/zone1970.tab"  # 备用：IANA时区列表
            ]

            all_timezones = None
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        if 'worldtimeapi' in url:
                            all_timezones = response.json()
                        else:
                            # 解析 IANA 格式
                            lines = response.text.split('\n')
                            all_timezones = []
                            for line in lines:
                                if line and not line.startswith('#'):
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        all_timezones.append(parts[1])
                        if all_timezones:
                            break
                except:
                    continue

            if all_timezones:
                get_timezone_list._cache = all_timezones
                get_timezone_list._cache_time = current_time
            else:
                # 使用内置常用时区列表
                all_timezones = [
                    'Asia/Shanghai', 'Asia/Tokyo', 'Asia/Seoul', 'Asia/Singapore', 'Asia/Hong_Kong',
                    'America/New_York', 'America/Los_Angeles', 'America/Chicago', 'America/Toronto',
                    'Europe/London', 'Europe/Paris', 'Europe/Berlin', 'Europe/Moscow',
                    'Australia/Sydney', 'Australia/Melbourne', 'Africa/Cairo', 'Africa/Johannesburg'
                ]
                get_timezone_list._cache = all_timezones
                get_timezone_list._cache_time = current_time

        except Exception as e:
            logger.error(f"获取时区列表失败: {str(e)}")
            all_timezones = get_timezone_list._cache or []

    all_timezones = get_timezone_list._cache or []

    if not all_timezones:
        return "无法获取时区列表，请检查网络连接"

    if region:
        filtered = [tz for tz in all_timezones if tz.startswith(region)]
        if filtered:
            result = f"🌍 {region}地区的时区（共{len(filtered)}个）：\n"
            for i, tz in enumerate(sorted(filtered)[:20], 1):
                result += f"{i}. {tz}\n"
            if len(filtered) > 20:
                result += f"... 还有{len(filtered) - 20}个时区"
            return result
        else:
            regions = set(tz.split('/')[0] for tz in all_timezones if '/' in tz)
            return f"未找到 {region} 地区的时区。可用地区：{', '.join(sorted(regions)[:10])}"
    else:
        regions = {}
        for tz in all_timezones:
            if '/' in tz:
                region_name = tz.split('/')[0]
                regions[region_name] = regions.get(region_name, 0) + 1

        result = f"🌍 支持的总时区数：{len(all_timezones)}\n\n"
        result += "📊 各地区时区统计：\n"
        for region_name, count in sorted(regions.items()):
            result += f"  • {region_name}: {count}个\n"

        result += "\n💡 常用时区示例：\n"
        result += "  • Asia/Shanghai (中国)\n"
        result += "  • America/New_York (美国东部)\n"
        result += "  • Europe/London (英国)\n"
        result += "  • Asia/Tokyo (日本)\n"
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
        '柏林': 'Europe/Berlin', '罗马': 'Europe/Rome', '莫斯科': 'Europe/Moscow',
        '悉尼': 'Australia/Sydney', '墨尔本': 'Australia/Melbourne', '迪拜': 'Asia/Dubai',
        '孟买': 'Asia/Kolkata'
    }

    city_lower = city_name.lower().strip()
    timezone = None

    # 直接匹配
    if city_name in city_timezone_map:
        timezone = city_timezone_map[city_name]
    elif city_lower in [k.lower() for k in city_timezone_map.keys()]:
        idx = [k.lower() for k in city_timezone_map.keys()].index(city_lower)
        timezone = list(city_timezone_map.values())[idx]
    else:
        # 模糊匹配
        for key in city_timezone_map:
            if key in city_name or city_name in key:
                timezone = city_timezone_map[key]
                break

    if not timezone:
        available = list(city_timezone_map.keys())[:15]
        return f"不支持城市 '{city_name}'。支持的城市示例：{', '.join(available)}"

    # 使用已有的 get_current_time_by_timezone 函数
    return get_current_time_by_timezone(timezone)


@tool(description="比较两个时区或城市的时间差异")
def compare_time(tz1: str, tz2: str) -> str:
    """比较两个时区或城市的时间差异"""
    # 简单的城市到时区映射
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
        # 尝试将常见城市名转换为时区
        for city, tz in city_map.items():
            if city in name or name in city:
                return tz
        return name

    timezone1 = normalize(tz1)
    timezone2 = normalize(tz2)

    try:
        # 获取两个时区的时间
        dt1, _, offset1 = get_time_from_multiple_apis(timezone1)
        dt2, _, offset2 = get_time_from_multiple_apis(timezone2)

        name1 = timezone1.split('/')[-1] if '/' in timezone1 else timezone1
        name2 = timezone2.split('/')[-1] if '/' in timezone2 else timezone2

        # 计算时差
        from dateutil import parser
        # 转换为 UTC 时间计算差值
        import pytz

        def parse_offset(offset_str):
            if not offset_str:
                return 0
            # 处理 +08:00 或 +0800 格式
            offset_str = offset_str.strip()
            if offset_str == 'Z' or offset_str == '+00:00':
                return 0
            sign = 1 if offset_str[0] == '+' else -1
            if ':' in offset_str:
                hours = int(offset_str[1:3])
                minutes = int(offset_str[4:6])
            else:
                hours = int(offset_str[1:3])
                minutes = int(offset_str[3:5]) if len(offset_str) >= 5 else 0
            return sign * (hours + minutes / 60)

        off1_hours = parse_offset(offset1)
        off2_hours = parse_offset(offset2)
        hour_diff = off2_hours - off1_hours

        return f"""
📊 时间对比结果
{'=' * 40}

📍 {name1} ({timezone1})
   🕐 {dt1.strftime('%H:%M:%S')}
   📅 {dt1.strftime('%Y-%m-%d')}
   🌍 {offset1 if offset1 else 'UTC+0'}

📍 {name2} ({timezone2})
   🕐 {dt2.strftime('%H:%M:%S')}
   📅 {dt2.strftime('%Y-%m-%d')}
   🌍 {offset2 if offset2 else 'UTC+0'}

{'=' * 40}
⏰ 时差：{name2} 比 {name1} {'早' if hour_diff > 0 else '晚'} {abs(hour_diff):.1f} 小时
        """.strip()

    except Exception as e:
        return f"比较失败：{str(e)}"


date_tools = [get_current_time_by_timezone, get_timezone_list, get_city_time, compare_time]
logger.info(f"成功加载工具: {[tool.name for tool in date_tools]}")