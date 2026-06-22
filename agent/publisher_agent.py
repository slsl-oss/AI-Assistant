"""
排版导出 Agent：主工作流 publisher 节点，组装完整 Markdown 报告
"""
from utils.logger_handler import logger


class PublisherAgent:
    """排版导出最终报告"""

    async def run(
        self, title: str, date: str, introduction: str, conclusion: str,
        table_of_contents: str, research_data: list[dict], sources: list[str], headers: dict,
    ) -> str:
        """主工作流 publisher 节点：组装完整 Markdown 报告"""
        parts = [
            f"# {title}", "", f"> 生成日期：{date}", f"> 本报告由 AI 自动生成，仅供参考",
            "", "---", "", "## 引言", "", introduction, "", "---", "",
            "## 目录", "", table_of_contents, "", "---", "",
        ]
        for i, rd in enumerate(research_data, 1):
            parts += [f"## {i}. {rd['section']}", "", rd["draft"], "", "---", ""]
        parts += ["## 结论", "", conclusion, "", "---", ""]
        if sources:
            parts.append("## 参考来源")
            parts.append("")
            for i, src in enumerate(sources, 1):
                parts.append(f"{i}. [{src}]({src})")
            parts.append("")

        report = "\n".join(parts)
        logger.info(f"[Publisher] 报告: {len(report)} 字符, {len(research_data)} 章节, {len(sources)} 来源")
        return report


publisher_agent = PublisherAgent()