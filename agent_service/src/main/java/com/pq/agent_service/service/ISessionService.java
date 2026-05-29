package com.pq.agent_service.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pq.agent_service.domain.po.Session;


public interface ISessionService extends IService<Session> {


    Session createSession(Long userId);

    void updateSession(String id);

    void deleteSessionMemory(String id, Long userId);
}
