package com.pq.agent_service.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pq.agent_service.domain.dto.MessageDTO;
import com.pq.agent_service.domain.po.Message;
import com.pq.agent_service.mapper.MessageMapper;
import com.pq.agent_service.service.IMessageService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

@Slf4j
@Service
@RequiredArgsConstructor
public class IMessageServiceImpl extends ServiceImpl<MessageMapper, Message> implements IMessageService {

    private final WebClient webClient;
    private final ExecutorService executorService = Executors.newCachedThreadPool();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public SseEmitter sendMessage(String sessionId, MessageDTO dto, Long userId) {
        SseEmitter emitter = new SseEmitter(300000L);
        AtomicReference<String> fullResponse = new AtomicReference<>("");
        AtomicReference<Boolean> isCompleted = new AtomicReference<>(false);

        executorService.execute(() -> {
            try {
                saveUserMessage(sessionId, dto);
                dto.setSessionId(sessionId);
                callPythonServiceStream(dto, emitter, fullResponse, isCompleted, userId);

            } catch (Exception e) {
                log.error("发送消息失败", e);
                safeComplete(emitter, isCompleted);
            }
        });

        return emitter;
    }

    private void saveUserMessage(String sessionId, MessageDTO dto) {
        Message userMessage = new Message();
        userMessage.setSessionId(sessionId);
        userMessage.setContent(dto.getContent());
        userMessage.setSenderType(1);
        userMessage.setMessageType(dto.getMessageType());
        userMessage.setCreateTime(LocalDateTime.now());
        this.save(userMessage);
    }

    private void callPythonServiceStream(MessageDTO dto, SseEmitter emitter, AtomicReference<String> fullResponse, AtomicReference<Boolean> isCompleted, Long userId) {
        StringBuilder responseBuilder = new StringBuilder();

        try {
            webClient.post()
                .uri("/sessions/messages/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                    "query", dto.getContent(),
                    "session_id", dto.getSessionId() != null ? dto.getSessionId() : "",
                    "user_id", String.valueOf(userId)
                ))
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {})
                .subscribe(
                    event -> {
                        try {
                            String data = event.data();
                            if (data == null || data.isEmpty()) return;
                            JsonNode jsonNode = objectMapper.readTree(data);
                            if (jsonNode.has("chunk")) {
                                String content = jsonNode.get("chunk").asText();
                                responseBuilder.append(content);
                                emitter.send(SseEmitter.event()
                                    .data(objectMapper.writeValueAsString(Map.of("chunk", content))));
                            } else if (jsonNode.has("thinking")) {
                                boolean thinking = jsonNode.get("thinking").asBoolean();
                                emitter.send(SseEmitter.event()
                                    .data(objectMapper.writeValueAsString(Map.of("thinking", thinking))));
                            } else if (jsonNode.has("done") && jsonNode.get("done").asBoolean()) {
                                String finalResponse = responseBuilder.toString();
                                fullResponse.set(finalResponse);
                                saveAiMessage(dto.getSessionId(), finalResponse);
                                emitter.send(SseEmitter.event()
                                    .data(objectMapper.writeValueAsString(Map.of("done", true))));
                                safeComplete(emitter, isCompleted);
                            }
                        } catch (Exception e) {
                            log.error("处理SSE事件失败", e);
                        }
                    },
                    error -> {
                        log.error("流式调用失败", error);
                        try {
                            emitter.send(SseEmitter.event()
                                .name("error")
                                .data("AI服务调用失败: " + error.getMessage()));
                        } catch (IOException e) {
                            log.error("发送错误消息失败", e);
                        }
                        safeComplete(emitter, isCompleted);
                    },
                    () -> {
                        safeComplete(emitter, isCompleted);
                    }
                );
        } catch (Exception e) {
            log.error("调用Python服务失败", e);
            throw new RuntimeException("AI服务暂时不可用，请稍后再试");
        }
    }
    
    private void safeComplete(SseEmitter emitter, AtomicReference<Boolean> isCompleted) {
        if (isCompleted.compareAndSet(false, true)) {
            try {
                emitter.complete();
            } catch (Exception e) {
                log.warn("SseEmitter已关闭或已完成");
            }
        }
    }

    private void saveAiMessage(String sessionId, String content) {
        if (sessionId == null || sessionId.isEmpty()) {
            return;
        }
        Message aiMessage = new Message();
        aiMessage.setSessionId(sessionId);
        aiMessage.setContent(content);
        aiMessage.setSenderType(2);
        aiMessage.setMessageType(1);
        aiMessage.setCreateTime(LocalDateTime.now());
        this.save(aiMessage);
    }
}
