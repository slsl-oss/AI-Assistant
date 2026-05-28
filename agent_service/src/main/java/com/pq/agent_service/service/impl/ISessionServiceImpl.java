package com.pq.agent_service.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pq.agent_service.domain.po.Session;
import com.pq.agent_service.mapper.SessionMapper;
import com.pq.agent_service.service.ISessionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Map;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Service
@RequiredArgsConstructor
public class ISessionServiceImpl extends ServiceImpl<SessionMapper, Session> implements ISessionService {

    private final WebClient webClient;

    @Override
    public Session createSession() {
        Session session = new Session();
        session.setCreateTime(LocalDateTime.now());
        session.setUpdateTime(LocalDateTime.now());
        session.setStatus(1);

        String sessionId = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
        session.setId(sessionId);


        //保存会话到数据库
        saveOrUpdate(session);

        return session;
    }

    @Override
    public void updateSession(String id) {
        Session session = getById(id);
        session.setUpdateTime(LocalDateTime.now());
        session.setStatus(1);

    }

    @Override
    public void deleteSessionMemory(String id) {
        try {
            webClient.delete()
                    .uri("/sessions/{sessionId}/memory", id)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .subscribe(
                            response -> log.info("成功删除会话记忆: sessionId={}, response={}", id, response),
                            error -> log.error("删除会话记忆失败: sessionId={},     error={}", id, error.getMessage())
                    );
        } catch (Exception e) {
            log.error("调用 Python 后端删除会话记忆异常: sessionId={}", id, e);
        }
    }

}
