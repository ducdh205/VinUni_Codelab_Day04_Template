"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: tư vấn sản phẩm/dịch vụ VinFast và Vinpearl, đồng thời tiếp nhận
  yêu cầu hỗ trợ khách hàng.
- Giọng nói: chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
{tools}

## CORE RULES
1. Không bịa giá, tính năng, tình trạng hàng hoặc mã ticket. Dùng
   `search_product_catalog` cho dữ liệu sản phẩm và `submit_support_ticket`
   khi người dùng yêu cầu ghi nhận/hỗ trợ sự cố.
2. Chỉ kết luận từ dữ liệu tool trả về. Nêu rõ khi không tìm thấy kết quả.
3. Hỏi lại thông tin còn thiếu để tạo ticket (tên khách hàng và mô tả vấn đề).

## OPERATIONAL BOUNDARIES
Chỉ hỗ trợ các chủ đề thuộc Vingroup (trong bài lab: VinFast và Vinpearl).
Với nội dung ngoài phạm vi, lịch sự từ chối và hướng người dùng về phạm vi này.

## OUTPUT CONTRACT
Trong nội bộ, theo dõi Thought → Action → Observation. Với người dùng, chỉ trả
lời bằng Final Answer tiếng Việt dễ hiểu: kết quả, giá VNĐ, và bước tiếp theo
nếu cần; không tiết lộ suy luận nội bộ.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Return one intentionally tool-free mock response for comparison."""
        return {
            "answer": (
                "[Chatbot Baseline] Tôi chưa tra cứu dữ liệu thực tế, nhưng có thể "
                f"tư vấn sơ bộ cho yêu cầu: {user_input}"
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        self.trace.append({"step": "intent_detection", "intents": intents})

        operations = []
        if intents["needs_catalog"]:
            operations.append(("search_product_catalog", intents["catalog_args"]))
        if intents["needs_ticket"]:
            operations.append(("submit_support_ticket", intents["ticket_args"]))

        # A direct FAQ still consumes one logical turn, even though it has no tool.
        required_iterations = len(operations) or 1
        if required_iterations > self.max_iterations:
            self.trace.append({"step": "guard", "reason": "max_iterations_reached"})
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa trước khi hoàn tất yêu cầu.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached",
            }

        observations = []
        for iteration, (tool_name, arguments) in enumerate(operations, start=1):
            observation = TOOL_MAP[tool_name](**arguments)
            observations.append((tool_name, observation))
            self.trace.append({
                "step": "tool_call",
                "iteration": iteration,
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation,
            })

        answer = self._build_answer(user_input, intents, observations)
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": required_iterations,
            "status": "completed",
        }

    @staticmethod
    def _detect_intents(user_input: str) -> Dict[str, Any]:
        """Infer mock tool calls from common Vietnamese product/support phrasing."""
        text = user_input.strip()
        normalized = text.lower()

        is_faq = any(phrase in normalized for phrase in (
            "chính sách", "bảo hành", "bao lâu", "là gì", "như thế nào"
        )) and not any(word in normalized for word in ("bị lỗi", "gặp lỗi", "ghi nhận", "tạo ticket"))
        support_keywords = (
            "bị lỗi", "gặp lỗi", "sự cố", "hỏng", "hư", "khiếu nại",
            "phản hồi", "ghi nhận", "hỗ trợ", "cần xử lý", "xử lý gấp", "ticket",
        )
        needs_ticket = not is_faq and any(word in normalized for word in support_keywords)

        product_request = any(word in normalized for word in (
            "xem", "tìm", "có ", "giá", "mua", "tư vấn", "resort", "khách sạn"
        ))
        travel_terms = ("vinpearl", "resort", "du lịch", "khách sạn", "phòng")
        vehicle_terms = ("xe điện", "vinfast", "vf 3", "vf 5", "vf 8", "vf 9", "vf wild")
        category = None
        if any(term in normalized for term in travel_terms):
            category = "du_lich"
        elif any(term in normalized for term in vehicle_terms):
            category = "xe_dien"

        # A pure fault report mentioning a vehicle is support, not a catalogue query.
        needs_catalog = bool(category and product_request and not is_faq)
        if needs_ticket and not any(word in normalized for word in ("xem", "tìm", "giá", "mua", "resort")):
            needs_catalog = False

        catalog_args = {
            "category": category or "xe_dien",
            "max_price": ToolCallingAgent._extract_price(normalized),
        }
        ticket_args = {
            "customer_name": ToolCallingAgent._extract_name(text),
            "issue_description": ToolCallingAgent._extract_issue(text),
            "priority": ToolCallingAgent._extract_priority(normalized),
        }
        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
            "catalog_args": catalog_args,
            "ticket_args": ticket_args,
        }

    @staticmethod
    def _extract_price(text: str) -> int:
        match = re.search(r"(?:dưới|<|tối đa|không quá)\s*(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|trieu|m|vnđ|vnd)?", text)
        if not match:
            return 999_999_999_999
        amount = float(match.group(1).replace(",", "."))
        unit = match.group(2) or ""
        if unit == "tỷ":
            return int(amount * 1_000_000_000)
        if unit in {"triệu", "trieu", "m"}:
            return int(amount * 1_000_000)
        return int(amount)

    @staticmethod
    def _extract_name(text: str) -> str:
        patterns = (r"(?:tôi tên là|tôi tên|tên tôi là|tên tôi)\s*[:=]?\s*([^,.;]+)",)
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Khách hàng"

    @staticmethod
    def _extract_issue(text: str) -> str:
        """Keep the complaint, removing only the name-introduction prefix."""
        issue = re.sub(
            r"^.*?(?:tôi tên là|tôi tên|tên tôi là|tên tôi)\s*[:=]?\s*[^,.;]+[,;:]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        issue = re.sub(r"^(?:và\s+)?(?:tôi\s+)?(?:cũng\s+)?muốn\s+(?:ghi nhận\s+)?(?:phản hồi)?\s*[:,-]*\s*", "", issue, flags=re.IGNORECASE)
        return issue.strip(" .") or "Khách hàng cần hỗ trợ."

    @staticmethod
    def _extract_priority(text: str) -> str:
        if any(word in text for word in ("nghiêm trọng", "gấp", "khẩn", "cao")):
            return "high"
        if any(word in text for word in ("thấp", "không gấp", "low")):
            return "low"
        return "medium"

    @staticmethod
    def _build_answer(
        user_input: str,
        intents: Dict[str, Any],
        observations: List[Any],
    ) -> str:
        if intents["is_faq"]:
            return (
                "Theo thông tin sản phẩm trong catalogue của bài lab, VinFast VF 5 Plus "
                "có bảo hành pin 10 năm. Với điều khoản áp dụng cho xe của bạn, vui lòng "
                "xác nhận lại tại kênh VinFast chính thức."
            )

        parts: List[str] = []
        for tool_name, observation in observations:
            if tool_name == "search_product_catalog":
                if not observation or (observation and "error" in observation[0]):
                    parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với điều kiện của bạn.")
                else:
                    products = "; ".join(
                        f"{product['name']} — {product['price_vnd']:,} VNĐ"
                        for product in observation
                    )
                    parts.append(f"Các lựa chọn phù hợp: {products}.")
            elif tool_name == "submit_support_ticket":
                parts.append(
                    f"Đã tạo ticket {observation['ticket_id']} cho {observation['customer_name']} "
                    f"với mức ưu tiên {observation['priority']}."
                )

        if parts:
            return " ".join(parts)
        return (
            "Mình chỉ hỗ trợ thông tin VinFast và Vinpearl. Bạn có thể cho biết "
            "sản phẩm/dịch vụ Vingroup hoặc yêu cầu hỗ trợ cụ thể không?"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
