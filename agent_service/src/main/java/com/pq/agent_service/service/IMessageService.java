package com.pq.agent_service.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pq.agent_service.domain.dto.MessageDTO;
import com.pq.agent_service.domain.po.Message;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;


public interface IMessageService extends IService<Message> {

    SseEmitter sendMessage(String id, MessageDTO dto);
}
