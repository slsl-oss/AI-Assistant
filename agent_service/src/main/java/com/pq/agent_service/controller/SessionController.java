package com.pq.agent_service.controller;

import com.pq.agent_service.domain.dto.MessageDTO;
import com.pq.agent_service.domain.po.Message;
import com.pq.agent_service.domain.po.Session;
import com.pq.agent_service.domain.vo.MessageVO;
import com.pq.agent_service.domain.vo.SessionVO;
import com.pq.agent_service.service.IMessageService;
import com.pq.agent_service.service.ISessionService;
import com.pq.agent_service.utils.BeanUtils;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.pq.agent_service.utils.UserContext;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Slf4j
@Api(tags = "会话相关接口")
@RestController
@RequestMapping("/sessions")
@RequiredArgsConstructor
public class SessionController {

    private final ISessionService sessionService;
    private final IMessageService messageService;
    private final WebClient webClient;
    @PostMapping
    @ApiOperation("创建新会话")
    public SessionVO createSession() {
        Long userId = UserContext.getUser().getId();
        Session session = sessionService.createSession(userId);
        return BeanUtils.copyBean(session, SessionVO.class);
    }

    @GetMapping
    @ApiOperation("获取所有会话列表")
    public List<SessionVO> listSessions() {
        Long userId = UserContext.getUser().getId();
        List<Session> sessions = sessionService.lambdaQuery()
            .eq(Session::getUserId, userId)
            .orderByDesc(Session::getCreateTime)
            .list();
        return BeanUtils.copyList(sessions, SessionVO.class);
    }

    @GetMapping("/{id}")
    @ApiOperation("获取单个会话")
    public SessionVO getSession(@PathVariable String id) {
        Session session = sessionService.getById(id);
        session.setStatus(1);
        if (session == null) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "会话不存在");
        }
        return BeanUtils.copyBean(session, SessionVO.class);
    }

    @DeleteMapping("/{id}")
    @ApiOperation("删除会话及其所有消息")
    public void deleteSession(@PathVariable String id) {
        Long userId = UserContext.getUser().getId();
        // 1. 删除数据库中的消息和会话
        messageService.lambdaUpdate()
            .eq(Message::getSessionId, id)
            .remove();
        sessionService.removeById(id);

        // 2. 调用 Python 后端删除 checkpointer 中的会话记忆
        sessionService.deleteSessionMemory(id, userId);
    }


    @GetMapping("/{id}/messages")
    @ApiOperation("获取会话的所有消息")
    public List<MessageVO> listMessages(@PathVariable String id) {
        List<Message> messages = messageService.lambdaQuery()
            .eq(Message::getSessionId, id)
            .orderByAsc(Message::getCreateTime)
            .list();
        return BeanUtils.copyList(messages, MessageVO.class);
    }

    @PostMapping("/{id}/messages/stream")
    @ApiOperation("发送消息并获取AI流式回复")
    public SseEmitter chatStream(@PathVariable String id, @RequestBody MessageDTO dto) {
        Long userId = UserContext.getUser().getId();
        sessionService.updateSession(id);
        return messageService.sendMessage(id, dto, userId);

    }

}
